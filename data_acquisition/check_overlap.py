#!/usr/bin/env python3
"""Check n-gram overlap between synthetic documents and their source case opinions.

Computes n-gram overlap to verify that LLM-generated documents are sufficiently
different from the real case text they were grounded on. High overlap would
indicate the model copied rather than generated novel content.

Layer A documents have a clear 1:1 source case (extracted from doc_id).
Layer B/C documents are compared against all cases in the corpus and the
maximum overlap with any single case is reported.

Usage:
    python data_acquisition/check_overlap.py \
        --corpus-dir corpus_text/legal_casebook \
        [--ngram-size 5] \
        [--threshold 15.0] \
        [--output corpus_text/legal_casebook/metadata/overlap_report.json]

Requires: tqdm
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
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
# N-gram overlap computation
# ---------------------------------------------------------------------------


def extract_ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    """Extract word-level n-grams from text."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def compute_overlap(synthetic_text: str, source_text: str, n: int = 5) -> dict:
    """Compute n-gram overlap between synthetic and source texts.

    Returns a dict with:
      - overlap_pct: % of synthetic n-grams found in source
      - synthetic_ngrams: total unique n-grams in synthetic doc
      - shared_ngrams: n-grams shared between synthetic and source
    """
    syn_ngrams = extract_ngrams(synthetic_text, n)
    src_ngrams = extract_ngrams(source_text, n)

    if not syn_ngrams:
        return {"overlap_pct": 0.0, "synthetic_ngrams": 0, "shared_ngrams": 0}

    shared = syn_ngrams & src_ngrams
    return {
        "overlap_pct": 100.0 * len(shared) / len(syn_ngrams),
        "synthetic_ngrams": len(syn_ngrams),
        "shared_ngrams": len(shared),
    }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_case_texts(corpus_dir: str, case_index: dict) -> dict[str, str]:
    """Load all case texts keyed by case_id."""
    texts = {}
    for case_meta in case_index.get("cases", []):
        case_id = case_meta["case_id"]
        file_path = os.path.join(corpus_dir, case_meta["file"])
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                texts[case_id] = f.read()
    return texts


def load_synthetic_docs(output_dir: str) -> dict[str, list[dict]]:
    """Load all synthetic documents from progress files, grouped by layer."""
    layers = {}
    for layer_name in ("a", "b", "c"):
        progress = os.path.join(output_dir, f"layer_{layer_name}_progress.json")
        if os.path.exists(progress):
            with open(progress) as f:
                docs = json.load(f)
            layers[layer_name.upper()] = docs
            logger.info(f"Layer {layer_name.upper()}: loaded {len(docs)} documents")
    return layers


def extract_case_id_from_doc_id(doc_id: str) -> str | None:
    """Extract the case_id from a Layer A doc_id.

    Layer A format: layerA_{case_id}_{doc_type}
    The case_id is a numeric string (e.g., '6218684').
    """
    match = re.match(r"layerA_(\d+)_", doc_id)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Overlap checks
# ---------------------------------------------------------------------------


def check_layer_a(
    docs: list[dict],
    case_texts: dict[str, str],
    ngram_size: int,
    threshold: float,
) -> list[dict]:
    """Check Layer A docs against their specific source cases."""
    results = []
    for doc in tqdm(docs, desc="Layer A overlap"):
        case_id = extract_case_id_from_doc_id(doc["doc_id"])
        if not case_id or case_id not in case_texts:
            results.append({
                "doc_id": doc["doc_id"],
                "doc_type": doc["doc_type"],
                "source_case_id": case_id,
                "overlap_pct": None,
                "status": "skipped",
                "reason": "source case not found",
            })
            continue

        overlap = compute_overlap(doc["content"], case_texts[case_id], ngram_size)
        flagged = overlap["overlap_pct"] > threshold
        results.append({
            "doc_id": doc["doc_id"],
            "doc_type": doc["doc_type"],
            "source_case_id": case_id,
            **overlap,
            "flagged": flagged,
            "status": "FLAGGED" if flagged else "ok",
        })
    return results


def check_layer_bc(
    docs: list[dict],
    case_texts: dict[str, str],
    ngram_size: int,
    threshold: float,
    layer_label: str,
) -> list[dict]:
    """Check Layer B/C docs against all cases, report max overlap."""
    # Pre-compute all case n-grams
    logger.info(f"Pre-computing n-grams for {len(case_texts)} cases...")
    case_ngrams = {
        cid: extract_ngrams(text, ngram_size)
        for cid, text in tqdm(case_texts.items(), desc="Case n-grams")
    }

    results = []
    for doc in tqdm(docs, desc=f"Layer {layer_label} overlap"):
        syn_ngrams = extract_ngrams(doc["content"], ngram_size)
        if not syn_ngrams:
            results.append({
                "doc_id": doc["doc_id"],
                "doc_type": doc["doc_type"],
                "max_overlap_pct": 0.0,
                "max_overlap_case": None,
                "status": "ok",
            })
            continue

        max_overlap = 0.0
        max_case_id = None
        for cid, src_ngrams in case_ngrams.items():
            shared = len(syn_ngrams & src_ngrams)
            pct = 100.0 * shared / len(syn_ngrams)
            if pct > max_overlap:
                max_overlap = pct
                max_case_id = cid

        flagged = max_overlap > threshold
        results.append({
            "doc_id": doc["doc_id"],
            "doc_type": doc["doc_type"],
            "max_overlap_pct": round(max_overlap, 2),
            "max_overlap_case": max_case_id,
            "synthetic_ngrams": len(syn_ngrams),
            "flagged": flagged,
            "status": "FLAGGED" if flagged else "ok",
        })
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(
    layer_results: dict[str, list[dict]],
    ngram_size: int,
    threshold: float,
) -> dict:
    """Build a summary report from per-document results."""
    summary: dict[str, Any] = {
        "ngram_size": ngram_size,
        "threshold_pct": threshold,
        "layers": {},
        "by_doc_type": {},
    }

    total_docs = 0
    total_flagged = 0

    for layer, results in layer_results.items():
        overlaps = []
        flagged = 0
        for r in results:
            pct = r.get("overlap_pct") or r.get("max_overlap_pct")
            if pct is not None:
                overlaps.append(pct)
            if r.get("flagged"):
                flagged += 1

        layer_summary = {
            "total_docs": len(results),
            "checked": len(overlaps),
            "flagged": flagged,
            "mean_overlap_pct": round(sum(overlaps) / len(overlaps), 2) if overlaps else 0,
            "max_overlap_pct": round(max(overlaps), 2) if overlaps else 0,
            "min_overlap_pct": round(min(overlaps), 2) if overlaps else 0,
        }
        summary["layers"][layer] = layer_summary
        total_docs += len(results)
        total_flagged += flagged

    # Aggregate by document type
    type_overlaps: dict[str, list[float]] = defaultdict(list)
    for results in layer_results.values():
        for r in results:
            pct = r.get("overlap_pct") or r.get("max_overlap_pct")
            if pct is not None:
                type_overlaps[r["doc_type"]].append(pct)

    for doc_type, overlaps in sorted(type_overlaps.items()):
        summary["by_doc_type"][doc_type] = {
            "count": len(overlaps),
            "mean_overlap_pct": round(sum(overlaps) / len(overlaps), 2),
            "max_overlap_pct": round(max(overlaps), 2),
            "min_overlap_pct": round(min(overlaps), 2),
        }

    summary["total_docs"] = total_docs
    summary["total_flagged"] = total_flagged
    summary["pass"] = total_flagged == 0

    return summary


def print_report(summary: dict, layer_results: dict[str, list[dict]]):
    """Print a human-readable overlap report."""
    print("\n" + "=" * 60)
    print(f"N-GRAM OVERLAP REPORT ({summary['ngram_size']}-grams)")
    print(f"Threshold: {summary['threshold_pct']}%")
    print("=" * 60)

    # Per-layer summary
    for layer, stats in summary["layers"].items():
        print(f"\n  Layer {layer}:")
        print(f"    Documents: {stats['total_docs']} ({stats['checked']} checked)")
        print(f"    Mean overlap: {stats['mean_overlap_pct']:.1f}%")
        print(f"    Range: {stats['min_overlap_pct']:.1f}% – {stats['max_overlap_pct']:.1f}%")
        if stats["flagged"]:
            print(f"    FLAGGED: {stats['flagged']} documents above threshold")

    # Per document type
    print(f"\n  By document type:")
    for doc_type, stats in summary["by_doc_type"].items():
        flag = " (!)" if stats["max_overlap_pct"] > summary["threshold_pct"] else ""
        print(f"    {doc_type:30s}  mean={stats['mean_overlap_pct']:5.1f}%  "
              f"max={stats['max_overlap_pct']:5.1f}%  (n={stats['count']}){flag}")

    # Flagged documents
    all_flagged = []
    for layer, results in layer_results.items():
        for r in results:
            if r.get("flagged"):
                all_flagged.append((layer, r))

    if all_flagged:
        print(f"\n  FLAGGED DOCUMENTS ({len(all_flagged)}):")
        for layer, r in all_flagged:
            pct = r.get("overlap_pct") or r.get("max_overlap_pct", 0)
            src = r.get("source_case_id") or r.get("max_overlap_case", "?")
            print(f"    [{layer}] {r['doc_id']}  overlap={pct:.1f}%  source={src}")

    # Overall
    print(f"\n  Overall: {summary['total_docs']} documents, "
          f"{summary['total_flagged']} flagged")
    status = "PASS" if summary["pass"] else "FAIL"
    print(f"  Status: {status}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Check n-gram overlap between synthetic and source documents."
    )
    parser.add_argument(
        "--corpus-dir", default="corpus_text/legal_casebook",
        help="Directory containing case files and metadata.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory containing augmented docs. "
             "Defaults to {corpus-dir}/augmented/.",
    )
    parser.add_argument(
        "--ngram-size", "-n", type=int, default=5,
        help="N-gram size for overlap computation (default: 5).",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=15.0,
        help="Flag documents with overlap above this %% (default: 15.0).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for overlap report JSON. "
             "Defaults to {corpus-dir}/metadata/overlap_report.json.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with non-zero code if any document is flagged.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.corpus_dir, "augmented")
    if args.output is None:
        args.output = os.path.join(args.corpus_dir, "metadata", "overlap_report.json")

    # Load case index
    index_path = os.path.join(args.corpus_dir, "metadata", "case_index.json")
    if not os.path.exists(index_path):
        sys.exit(f"Case index not found: {index_path}")
    with open(index_path) as f:
        case_index = json.load(f)

    # Load case texts
    logger.info("Loading case texts...")
    case_texts = load_case_texts(args.corpus_dir, case_index)
    logger.info(f"Loaded {len(case_texts)} cases")

    # Load synthetic documents
    synthetic = load_synthetic_docs(args.output_dir)
    if not synthetic:
        sys.exit("No synthetic documents found. Run generate_augmentation.py first.")

    # Run overlap checks
    layer_results: dict[str, list[dict]] = {}

    if "A" in synthetic:
        logger.info("Checking Layer A overlap...")
        layer_results["A"] = check_layer_a(
            synthetic["A"], case_texts, args.ngram_size, args.threshold,
        )

    if "B" in synthetic:
        logger.info("Checking Layer B overlap...")
        layer_results["B"] = check_layer_bc(
            synthetic["B"], case_texts, args.ngram_size, args.threshold, "B",
        )

    if "C" in synthetic:
        logger.info("Checking Layer C overlap...")
        layer_results["C"] = check_layer_bc(
            synthetic["C"], case_texts, args.ngram_size, args.threshold, "C",
        )

    # Build and print report
    summary = build_report(layer_results, args.ngram_size, args.threshold)
    print_report(summary, layer_results)

    # Save report
    report = {
        "summary": summary,
        "details": {
            layer: results for layer, results in layer_results.items()
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {args.output}")

    if args.strict and not summary["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

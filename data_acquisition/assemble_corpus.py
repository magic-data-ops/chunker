#!/usr/bin/env python3
"""Phase 4: Assemble the final legal casebook corpus from real cases + augmented docs.

Merges all layers (real case opinions + litigation documents + legislative materials
+ secondary sources) into a single legal_casebook_complete.txt file with document
type markers. Documents are interleaved by topic cluster in a deliberate
non-chronological order to create natural temporal_ordering challenges.

Usage:
    python data_acquisition/assemble_corpus.py \
        --corpus-dir corpus_text/legal_casebook \
        --output corpus_text/legal_casebook/legal_casebook_complete.txt

Requires: tqdm
"""

import argparse
import json
import logging
import os
import random
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
# Constants
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4.0

# Map augmented doc types to display labels
DOC_TYPE_LABELS = {
    # Real cases
    "court_opinion": "COURT_OPINION",
    # Layer A
    "plaintiff_brief": "PLAINTIFF_BRIEF",
    "defendant_brief": "DEFENDANT_BRIEF",
    "motion_summary_judgment": "MOTION_SUMMARY_JUDGMENT",
    "motion_discovery": "MOTION_DISCOVERY",
    "settlement_agreement": "SETTLEMENT_AGREEMENT",
    "court_order": "COURT_ORDER",
    # Layer B
    "statutory_text": "STATUTORY_TEXT",
    "committee_report": "COMMITTEE_REPORT",
    "floor_debate": "FLOOR_DEBATE",
    "regulatory_guidance": "REGULATORY_GUIDANCE",
    # Layer C
    "law_review_article": "LAW_REVIEW_ARTICLE",
    "practice_guide": "PRACTICE_GUIDE",
    "treatise_section": "TREATISE_SECTION",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Document collection
# ---------------------------------------------------------------------------


def load_real_cases(corpus_dir: str) -> list[dict]:
    """Load real case opinions from case files."""
    index_path = os.path.join(corpus_dir, "metadata", "case_index.json")
    if not os.path.exists(index_path):
        logger.warning(f"Case index not found: {index_path}")
        return []

    with open(index_path) as f:
        case_index = json.load(f)

    documents = []
    for case_meta in tqdm(case_index.get("cases", []), desc="Loading real cases", unit="case"):
        file_path = os.path.join(corpus_dir, case_meta["file"])
        if not os.path.exists(file_path):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        documents.append({
            "doc_type": "court_opinion",
            "case_id": case_meta.get("case_id", ""),
            "case_name": case_meta.get("case_name", "Unknown"),
            "court": case_meta.get("court", ""),
            "date": case_meta.get("date", ""),
            "jurisdiction": case_meta.get("jurisdiction", ""),
            "topic": case_meta.get("topic", "general"),
            "citations": case_meta.get("citations", []),
            "text": text,
        })

    logger.info(f"Loaded {len(documents)} real case opinions")
    return documents


def load_augmented_docs(corpus_dir: str) -> list[dict]:
    """Load augmented documents from all layers."""
    augmented_dir = os.path.join(corpus_dir, "augmented")
    documents = []

    for layer in ["a", "b", "c"]:
        progress_path = os.path.join(augmented_dir, f"layer_{layer}_progress.json")
        if not os.path.exists(progress_path):
            logger.info(f"No Layer {layer.upper()} progress file found, skipping")
            continue

        with open(progress_path) as f:
            try:
                docs = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {progress_path}")
                continue

        for doc in docs:
            content = doc.get("content", "")
            if not content or len(content) < 100:
                continue

            # Try to infer topic from doc_id
            doc_id = doc.get("doc_id", "")
            topic = "general"
            # Layer A: doc_id = "layerA_{case_id}_{doc_type}" — lookup case topic
            if doc_id.startswith("layerA_"):
                parts = doc_id.split("_", 2)
                if len(parts) >= 2:
                    topic = f"case_{parts[1]}"  # Will be resolved later
            # Layer C: doc_id includes topic
            elif doc_id.startswith("layerC_"):
                # Extract topic between "layerC_" and the doc_type
                match = re.match(r"layerC_(.+?)_(law_review|practice_guide|treatise)", doc_id)
                if match:
                    topic = match.group(1).replace("_", " ")

            documents.append({
                "doc_type": doc.get("doc_type", "unknown"),
                "doc_id": doc_id,
                "topic": topic,
                "text": content,
                "layer": layer.upper(),
            })

    logger.info(f"Loaded {len(documents)} augmented documents")
    return documents


# ---------------------------------------------------------------------------
# Topic-based interleaving
# ---------------------------------------------------------------------------


def assign_topics(real_cases: list[dict], augmented_docs: list[dict],
                  case_index: dict) -> list[dict]:
    """Assign consistent topics and prepare all documents for interleaving.

    Returns a list of all documents with normalized topic assignments.
    """
    # Build case_id → topic lookup from case_index
    case_topic_map = {}
    for case in case_index.get("cases", []):
        case_topic_map[case.get("case_id", "")] = case.get("topic", "general")

    all_docs = []

    # Real cases: already have topics
    for doc in real_cases:
        all_docs.append(doc)

    # Augmented docs: resolve topic from case_id if needed
    for doc in augmented_docs:
        topic = doc.get("topic", "general")
        # Resolve case_id-based topics for Layer A
        if topic.startswith("case_"):
            case_id = topic[5:]
            topic = case_topic_map.get(case_id, "general")
            doc["topic"] = topic
        all_docs.append(doc)

    return all_docs


def interleave_by_topic(documents: list[dict], seed: int = 42) -> list[dict]:
    """Interleave documents by topic cluster in non-chronological order.

    Groups documents by topic, then interleaves topic groups. Within each
    topic group, shuffles to break chronological order.
    """
    rng = random.Random(seed)

    # Group by topic
    topic_groups: dict[str, list[dict]] = defaultdict(list)
    for doc in documents:
        topic = doc.get("topic", "general")
        topic_groups[topic].append(doc)

    # Shuffle within each topic group (breaks chronological order)
    for topic, group in topic_groups.items():
        rng.shuffle(group)

    # Round-robin interleave across topics
    topics = sorted(topic_groups.keys())
    rng.shuffle(topics)  # Randomize topic order too

    result = []
    topic_iterators = {t: iter(topic_groups[t]) for t in topics}
    active_topics = list(topics)

    while active_topics:
        remaining = []
        for topic in active_topics:
            try:
                doc = next(topic_iterators[topic])
                result.append(doc)
                remaining.append(topic)
            except StopIteration:
                pass
        active_topics = remaining

    return result


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_final_corpus(documents: list[dict], output_path: str) -> dict:
    """Write the final assembled corpus file with document type markers.

    Returns statistics about the assembled corpus.
    """
    total = len(documents)
    total_chars = 0
    type_counts: dict[str, int] = defaultdict(int)
    topic_counts: dict[str, int] = defaultdict(int)
    jurisdiction_counts: dict[str, int] = defaultdict(int)

    # Write to temp file first, then rename atomically
    tmp_path = output_path + ".tmp"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(tmp_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(tqdm(documents, desc="Assembling corpus", unit="doc")):
            doc_type = doc.get("doc_type", "unknown")
            type_label = DOC_TYPE_LABELS.get(doc_type, doc_type.upper())

            # Build document header
            header_parts = [
                f"=== DOCUMENT {i + 1}/{total} | TYPE: {type_label} ===",
            ]

            # Add metadata based on document type
            if doc_type == "court_opinion":
                header_parts.extend([
                    f"CASE_ID: {doc.get('case_id', '')}",
                    f"CASE_NAME: {doc.get('case_name', '')}",
                    f"COURT: {doc.get('court', '')}",
                    f"DATE: {doc.get('date', '')}",
                    f"JURISDICTION: {doc.get('jurisdiction', '')}",
                    f"TOPIC: {doc.get('topic', '')}",
                    f"CITATIONS: {'; '.join(doc.get('citations', []))}",
                ])
                jurisdiction_counts[doc.get("jurisdiction", "unknown")] += 1
            else:
                header_parts.extend([
                    f"DOC_ID: {doc.get('doc_id', '')}",
                    f"TOPIC: {doc.get('topic', '')}",
                    f"LAYER: {doc.get('layer', '')}",
                ])

            header_parts.append("---")
            header = "\n".join(header_parts) + "\n"

            text = doc.get("text", "")
            full_entry = header + text + "\n\n"
            f.write(full_entry)
            total_chars += len(full_entry)

            type_counts[type_label] += 1
            topic_counts[doc.get("topic", "general")] += 1

    # Atomic rename
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(tmp_path, output_path)

    total_tokens = int(total_chars / CHARS_PER_TOKEN)

    stats = {
        "total_documents": total,
        "total_characters": total_chars,
        "total_tokens_estimate": total_tokens,
        "document_type_counts": dict(type_counts),
        "topic_counts": dict(topic_counts),
        "jurisdiction_counts": dict(jurisdiction_counts),
        "output_path": output_path,
        "output_size_mb": round(total_chars / (1024 * 1024), 1),
    }

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Assemble final legal casebook corpus from all layers."
    )
    parser.add_argument(
        "--corpus-dir", default="corpus_text/legal_casebook",
        help="Directory containing case files, metadata, and augmented docs.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for the assembled corpus. "
             "Defaults to {corpus-dir}/legal_casebook_complete.txt.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for interleaving order.",
    )
    parser.add_argument(
        "--skip-augmented", action="store_true",
        help="Only include real case opinions (skip augmented documents).",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.corpus_dir, "legal_casebook_complete.txt")

    # Load case index
    index_path = os.path.join(args.corpus_dir, "metadata", "case_index.json")
    if not os.path.exists(index_path):
        sys.exit(f"Case index not found: {index_path}\nRun download_cap.py first.")
    with open(index_path) as f:
        case_index = json.load(f)

    # Load documents
    real_cases = load_real_cases(args.corpus_dir)
    augmented_docs = [] if args.skip_augmented else load_augmented_docs(args.corpus_dir)

    if not real_cases and not augmented_docs:
        sys.exit("No documents found. Run download_cap.py and generate_augmentation.py first.")

    # Assign consistent topics
    all_docs = assign_topics(real_cases, augmented_docs, case_index)
    logger.info(f"Total documents to assemble: {len(all_docs)}")

    # Interleave by topic in non-chronological order
    logger.info("Interleaving documents by topic cluster...")
    ordered_docs = interleave_by_topic(all_docs, seed=args.seed)

    # Assemble final corpus
    logger.info("Writing final corpus file...")
    stats = assemble_final_corpus(ordered_docs, args.output)

    # Save assembly stats
    stats_path = os.path.join(args.corpus_dir, "metadata", "assembly_stats.json")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 60)
    print("CORPUS ASSEMBLY COMPLETE")
    print("=" * 60)
    print(f"Total documents: {stats['total_documents']}")
    print(f"Total characters: {stats['total_characters']:,}")
    print(f"Estimated tokens: {stats['total_tokens_estimate']:,}")
    print(f"File size: {stats['output_size_mb']} MB")
    print(f"\nDocument types:")
    for dtype, count in sorted(stats["document_type_counts"].items()):
        print(f"  {dtype}: {count}")
    print(f"\nTopics:")
    for topic, count in sorted(stats["topic_counts"].items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count}")
    if stats["jurisdiction_counts"]:
        print(f"\nJurisdictions (real cases only):")
        for jur, count in sorted(stats["jurisdiction_counts"].items()):
            print(f"  {jur}: {count}")
    print(f"\nOutput: {stats['output_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

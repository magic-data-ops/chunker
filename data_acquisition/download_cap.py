#!/usr/bin/env python3
"""Phase 1: Download and filter case law from static.case.law (Harvard CAP).

Downloads case JSON files from https://static.case.law/, filters by jurisdiction
and topic keywords, extracts plain text with structured metadata, and outputs
individual case files plus a consolidated case_index.json.

Usage:
    python data_acquisition/download_cap.py \
        --output-dir corpus_text/legal_casebook \
        --reporters us f3d ny3d sw3d \
        --topics "contract law" "employment law" "environmental regulation" "intellectual property" \
        --date-start 1950 --date-end 2025 \
        --target-tokens 150000000

Requires: requests, tqdm
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")

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

STATIC_BASE = "https://static.case.law"

CHARS_PER_TOKEN = 4.0

# Reporter slugs grouped by target jurisdiction
REPORTER_GROUPS = {
    "us-supreme-court": ["us"],
    "federal-circuit": ["f2d", "f3d"],
    "ny": ["ny", "ny-2d", "ny3d"],
    "tx": ["sw2d", "sw3d", "tex"],
}

# Court name patterns for filtering circuit cases
CIRCUIT_COURT_PATTERNS = {
    "9th-circuit": re.compile(r"ninth circuit|9th cir", re.IGNORECASE),
    "5th-circuit": re.compile(r"fifth circuit|5th cir", re.IGNORECASE),
}

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "contract law": [
        "breach of contract", "consideration", "specific performance",
        "damages", "contractual obligation", "warranty", "indemnification",
        "liquidated damages", "promissory estoppel", "parol evidence",
        "unconscionability", "good faith", "material breach",
    ],
    "employment law": [
        "wrongful termination", "discrimination", "harassment",
        "Title VII", "ADA", "FMLA", "at-will employment",
        "wage", "overtime", "collective bargaining", "arbitration",
        "non-compete", "whistleblower", "retaliation",
    ],
    "environmental regulation": [
        "Clean Air Act", "Clean Water Act", "CERCLA", "Superfund",
        "EPA", "environmental impact", "NEPA", "pollution",
        "hazardous waste", "emission", "contamination", "remediation",
    ],
    "intellectual property": [
        "patent", "trademark", "copyright", "trade secret",
        "infringement", "prior art", "fair use", "licensing",
        "intellectual property", "Lanham Act", "DMCA",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _atomic_save(data: Any, path: str) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)


def _estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def _text_matches_topics(text: str, topics: list[str]) -> bool:
    text_lower = text.lower()
    for topic in topics:
        keywords = TOPIC_KEYWORDS.get(topic, [topic])
        if any(kw.lower() in text_lower for kw in keywords):
            return True
    return False


def _matched_topic(text: str, topics: list[str]) -> str:
    for t in topics:
        if _text_matches_topics(text, [t]):
            return t
    return "general"


def _extract_citations(text: str) -> list[str]:
    patterns = [
        r"\d+\s+(?:U\.S\.|S\.\s*Ct\.|L\.\s*Ed\.|F\.\d*d?|F\.\s*Supp\.\s*\d*d?|"
        r"Cal\.\s*\d*(?:st|d|th)?|N\.Y\.\s*\d*d?|S\.W\.\s*\d*d?|"
        r"N\.E\.\s*\d*d?|N\.W\.\s*\d*d?|S\.E\.\s*\d*d?|P\.\s*\d*d?|"
        r"So\.\s*\d*d?|A\.\s*\d*d?)\s+\d+",
    ]
    citations = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            citations.add(match.group().strip())
    return sorted(citations)


def _extract_judges(text: str) -> list[str]:
    judges = []
    patterns = [
        r"(?:Justice|Judge|Chief Justice)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"([A-Z][a-z]+),\s+(?:J\.|C\.J\.|JJ\.)",
    ]
    seen = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text[:3000]):
            name = match.group(1).strip()
            if name not in seen and len(name) > 2:
                judges.append(name)
                seen.add(name)
    return judges


# ---------------------------------------------------------------------------
# static.case.law download
# ---------------------------------------------------------------------------


def list_volumes(reporter: str) -> list[str]:
    """Get list of volume numbers for a reporter from static.case.law."""
    url = f"{STATIC_BASE}/{reporter}/VolumesMetadata.json"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Each volume entry has a "volume_number" field
        return [str(v.get("volume_number", v.get("volume_folder", ""))) for v in data]
    except Exception as e:
        logger.warning(f"Could not list volumes for {reporter}: {e}")
        return []


def download_volume_cases(reporter: str, volume: str,
                          session: requests.Session) -> list[dict]:
    """Download and parse all case JSONs from a volume zip."""
    zip_url = f"{STATIC_BASE}/{reporter}/{volume}.zip"
    try:
        resp = session.get(zip_url, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.debug(f"  Could not download {zip_url}: {e}")
        return []

    cases_raw = []
    try:
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            for name in zf.namelist():
                # Case files are in cases/ subfolder or at root matching pattern
                if "/cases/" in name and name.endswith(".json"):
                    try:
                        with zf.open(name) as cf:
                            case_data = json.loads(cf.read())
                        cases_raw.append(case_data)
                    except (json.JSONDecodeError, KeyError):
                        continue
    except zipfile.BadZipFile:
        logger.debug(f"  Bad zip for {reporter}/{volume}")
        return []

    return cases_raw


def download_volume_cases_direct(reporter: str, volume: str,
                                 session: requests.Session) -> list[dict]:
    """Download cases via CasesMetadata.json + individual case files.

    Fallback when zip download fails.
    """
    meta_url = f"{STATIC_BASE}/{reporter}/{volume}/CasesMetadata.json"
    try:
        resp = session.get(meta_url, timeout=30)
        resp.raise_for_status()
        cases_meta = resp.json()
    except Exception:
        return []

    cases_raw = []
    for cm in cases_meta:
        fname = cm.get("file_name", "")
        if not fname:
            continue
        case_url = f"{STATIC_BASE}/{reporter}/{volume}/cases/{fname}"
        try:
            resp = session.get(case_url, timeout=30)
            resp.raise_for_status()
            cases_raw.append(resp.json())
        except Exception:
            continue
        time.sleep(0.1)  # Rate limit courtesy

    return cases_raw


def parse_case_json(case_data: dict, reporter: str, topics: list[str],
                    date_start: int, date_end: int,
                    circuit_filter: str | None = None) -> dict | None:
    """Parse a single case JSON from static.case.law into our format.

    Returns None if the case doesn't match filters.
    """
    # Date filter
    decision_date = case_data.get("decision_date", "")
    if decision_date:
        try:
            year = int(decision_date[:4])
            if year < date_start or year > date_end:
                return None
        except (ValueError, IndexError):
            pass

    # Extract opinion text
    casebody = case_data.get("casebody", {})
    if isinstance(casebody, dict):
        opinions = casebody.get("opinions", [])
        if not opinions:
            # Try nested data structure
            opinions = casebody.get("data", {}).get("opinions", [])
    else:
        opinions = []

    opinion_text = "\n\n".join(
        op.get("text", "") for op in opinions if isinstance(op, dict)
    )
    if not opinion_text or len(opinion_text) < 500:
        return None

    # Court/circuit filter for federal reporters
    if circuit_filter:
        court_name = ""
        court_obj = case_data.get("court", {})
        if isinstance(court_obj, dict):
            court_name = court_obj.get("name", "")
        else:
            court_name = str(court_obj)
        pattern = CIRCUIT_COURT_PATTERNS.get(circuit_filter)
        if pattern and not pattern.search(court_name):
            return None

    # Topic filter
    if not _text_matches_topics(opinion_text, topics):
        return None

    # Build case record
    case_id = str(case_data.get("id", ""))
    court_obj = case_data.get("court", {})
    court_name = court_obj.get("name", "Unknown") if isinstance(court_obj, dict) else str(court_obj)

    citations = []
    for c in case_data.get("citations", []):
        if isinstance(c, dict):
            citations.append(c.get("cite", ""))
        elif isinstance(c, str):
            citations.append(c)

    # Extract cites_to for richer citation graph
    cites_to_ids = []
    for ct in case_data.get("cites_to", []):
        if isinstance(ct, dict):
            for cid in ct.get("case_ids", []):
                cites_to_ids.append(str(cid))

    jurisdiction = ""
    jur_obj = case_data.get("jurisdiction", {})
    if isinstance(jur_obj, dict):
        jurisdiction = jur_obj.get("name_long", jur_obj.get("name", ""))
    else:
        jurisdiction = str(jur_obj)

    return {
        "_cap_id": case_id,
        "case_name": case_data.get("name", "Unknown"),
        "name_abbreviation": case_data.get("name_abbreviation", ""),
        "court": court_name,
        "jurisdiction": jurisdiction,
        "reporter": reporter,
        "date": decision_date,
        "citations": [c for c in citations if c],
        "cites_to_ids": cites_to_ids,
        "topic": _matched_topic(opinion_text, topics),
        "opinion_text": opinion_text,
        "extracted_citations": _extract_citations(opinion_text),
        "judges": _extract_judges(opinion_text),
    }


def fetch_reporter(
    reporter: str,
    topics: list[str],
    date_start: int,
    date_end: int,
    max_cases: int,
    existing_ids: set[str],
    circuit_filter: str | None = None,
) -> list[dict]:
    """Fetch all matching cases from a reporter on static.case.law."""
    logger.info(f"  Fetching volumes for reporter: {reporter}")
    volumes = list_volumes(reporter)
    if not volumes:
        logger.warning(f"  No volumes found for {reporter}")
        return []

    logger.info(f"  {reporter}: {len(volumes)} volumes")

    session = requests.Session()
    session.headers["User-Agent"] = "legal-casebook-corpus/1.0 (research)"

    cases = []
    for vol in tqdm(volumes, desc=f"  {reporter}", unit="vol"):
        if len(cases) >= max_cases:
            break

        # Try zip first, fall back to direct
        raw_cases = download_volume_cases(reporter, vol, session)
        if not raw_cases:
            raw_cases = download_volume_cases_direct(reporter, vol, session)

        for case_data in raw_cases:
            case_id = str(case_data.get("id", ""))
            if case_id in existing_ids:
                continue

            parsed = parse_case_json(
                case_data, reporter, topics, date_start, date_end,
                circuit_filter=circuit_filter,
            )
            if parsed:
                cases.append(parsed)
                existing_ids.add(case_id)

                if len(cases) >= max_cases:
                    break

        time.sleep(0.2)  # Rate limit

    logger.info(f"  {reporter}: {len(cases)} cases matched")
    return cases


# ---------------------------------------------------------------------------
# Output (unchanged from original)
# ---------------------------------------------------------------------------


def write_case_files(cases: list[dict], output_dir: str) -> dict:
    cases_dir = os.path.join(output_dir, "cases")
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(cases_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    case_index: dict[str, Any] = {
        "total_cases": 0,
        "total_tokens_estimate": 0,
        "jurisdictions": defaultdict(int),
        "topics": defaultdict(int),
        "date_range": {"earliest": "", "latest": ""},
        "cases": [],
    }

    earliest = "9999"
    latest = "0000"

    for case in tqdm(cases, desc="Writing case files", unit="case"):
        jur_slug = re.sub(r"[^a-z0-9_-]", "_", case.get("reporter", "unknown").lower())
        jur_dir = os.path.join(cases_dir, jur_slug)
        os.makedirs(jur_dir, exist_ok=True)

        case_id = case["_cap_id"]
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(case_id))
        filepath = os.path.join(jur_dir, f"{safe_id}.txt")

        citations_str = "; ".join(c for c in case.get("citations", []) if c)
        judges_str = ", ".join(case.get("judges", []))
        header = (
            f"CASE_ID: {case_id}\n"
            f"CASE_NAME: {case['case_name']}\n"
            f"COURT: {case['court']}\n"
            f"DATE: {case['date']}\n"
            f"CITATIONS: {citations_str}\n"
            f"JUDGES: {judges_str}\n"
            f"TOPIC: {case['topic']}\n"
            f"JURISDICTION: {case['jurisdiction']}\n"
            f"---\n"
        )

        full_text = header + case["opinion_text"]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_text)

        tokens = _estimate_tokens(full_text)
        case_index["total_tokens_estimate"] += tokens
        case_index["jurisdictions"][case["jurisdiction"]] += 1
        case_index["topics"][case["topic"]] += 1

        if case["date"] and case["date"] < earliest:
            earliest = case["date"]
        if case["date"] and case["date"] > latest:
            latest = case["date"]

        case_index["cases"].append({
            "case_id": case_id,
            "case_name": case["case_name"],
            "name_abbreviation": case.get("name_abbreviation", ""),
            "court": case["court"],
            "jurisdiction": case["jurisdiction"],
            "reporter": case.get("reporter", ""),
            "date": case["date"],
            "citations": case.get("citations", []),
            "cites_to_ids": case.get("cites_to_ids", []),
            "extracted_citations": case.get("extracted_citations", []),
            "judges": case.get("judges", []),
            "topic": case["topic"],
            "file": os.path.relpath(filepath, output_dir),
            "token_estimate": tokens,
        })

    case_index["total_cases"] = len(cases)
    case_index["date_range"] = {"earliest": earliest, "latest": latest}
    case_index["jurisdictions"] = dict(case_index["jurisdictions"])
    case_index["topics"] = dict(case_index["topics"])

    index_path = os.path.join(metadata_dir, "case_index.json")
    _atomic_save(case_index, index_path)
    logger.info(f"Wrote case_index.json: {case_index['total_cases']} cases, "
                f"~{case_index['total_tokens_estimate']:,} tokens")
    return case_index


def assemble_base_corpus(cases: list[dict], output_dir: str) -> str:
    total = len(cases)
    output_path = os.path.join(output_dir, "legal_casebook_base.txt")
    total_chars = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, case in enumerate(tqdm(cases, desc="Assembling base corpus", unit="case")):
            case_id = case["_cap_id"]
            citations_str = "; ".join(c for c in case.get("citations", []) if c)
            judges_str = ", ".join(case.get("judges", []))
            header = (
                f"=== CASE {i + 1}/{total} ===\n"
                f"CASE_ID: {case_id}\n"
                f"CASE_NAME: {case['case_name']}\n"
                f"COURT: {case['court']}\n"
                f"DATE: {case['date']}\n"
                f"CITATIONS: {citations_str}\n"
                f"JUDGES: {judges_str}\n"
                f"---\n"
            )
            text = header + case["opinion_text"] + "\n\n"
            f.write(text)
            total_chars += len(text)

    total_tokens = int(total_chars / CHARS_PER_TOKEN)
    logger.info(f"Base corpus: {output_path} ({total_chars:,} chars, ~{total_tokens:,} tokens)")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download and filter case law from static.case.law."
    )
    parser.add_argument("--output-dir", default="corpus_text/legal_casebook")
    parser.add_argument(
        "--reporters", nargs="+",
        default=["us", "f3d", "f2d", "ny3d", "ny-2d", "sw3d", "sw2d"],
        help="Reporter slugs from static.case.law to download.",
    )
    parser.add_argument(
        "--topics", nargs="+",
        default=["contract law", "employment law",
                 "environmental regulation", "intellectual property"],
    )
    parser.add_argument("--date-start", type=int, default=1950)
    parser.add_argument("--date-end", type=int, default=2025)
    parser.add_argument("--target-tokens", type=int, default=150_000_000)
    parser.add_argument("--max-per-reporter", type=int, default=5000)
    parser.add_argument(
        "--circuit-filter", nargs="*", default=[],
        help="For federal reporters (f2d/f3d), filter to specific circuits. "
             "E.g. --circuit-filter 9th-circuit 5th-circuit",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resume support
    existing_ids: set[str] = set()
    index_path = os.path.join(args.output_dir, "metadata", "case_index.json")
    if args.resume and os.path.exists(index_path):
        with open(index_path) as f:
            existing_index = json.load(f)
        existing_ids = {c["case_id"] for c in existing_index.get("cases", [])}
        logger.info(f"Resuming: {len(existing_ids)} cases already downloaded")

    all_cases: list[dict] = []
    target_chars = int(args.target_tokens * CHARS_PER_TOKEN)
    total_chars = 0

    # Determine circuit filters for federal reporters
    circuit_filters = set(args.circuit_filter) if args.circuit_filter else None

    for reporter in args.reporters:
        logger.info(f"Processing reporter: {reporter}")

        # For federal reporters, optionally filter by circuit
        if reporter in ("f2d", "f3d", "f-appx") and circuit_filters:
            for circuit in circuit_filters:
                logger.info(f"  Filtering for {circuit}")
                cases = fetch_reporter(
                    reporter, args.topics, args.date_start, args.date_end,
                    args.max_per_reporter, existing_ids,
                    circuit_filter=circuit,
                )
                for case in cases:
                    all_cases.append(case)
                    total_chars += len(case.get("opinion_text", ""))
        else:
            cases = fetch_reporter(
                reporter, args.topics, args.date_start, args.date_end,
                args.max_per_reporter, existing_ids,
            )
            for case in cases:
                all_cases.append(case)
                total_chars += len(case.get("opinion_text", ""))

        logger.info(f"  Running total: {len(all_cases)} cases, ~{int(total_chars / CHARS_PER_TOKEN):,} tokens")

        if total_chars >= target_chars:
            logger.info("Target token count reached.")
            break

    if not all_cases:
        logger.warning("No cases downloaded.")
        if not existing_ids:
            sys.exit(1)
        logger.info("Using previously downloaded cases only.")

    # Sort by date
    all_cases.sort(key=lambda c: c.get("date", ""))

    # Write outputs
    logger.info("Writing individual case files...")
    case_index = write_case_files(all_cases, args.output_dir)

    logger.info("Assembling base corpus file...")
    corpus_path = assemble_base_corpus(all_cases, args.output_dir)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Total cases: {case_index['total_cases']}")
    print(f"Estimated tokens: {case_index['total_tokens_estimate']:,}")
    print(f"Jurisdictions: {json.dumps(case_index['jurisdictions'], indent=2)}")
    print(f"Topics: {json.dumps(case_index['topics'], indent=2)}")
    print(f"Date range: {case_index['date_range']['earliest']} - {case_index['date_range']['latest']}")
    print(f"Base corpus: {corpus_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

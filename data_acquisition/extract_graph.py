#!/usr/bin/env python3
"""Phase 2: Extract entity & citation graph from the base legal casebook corpus.

Parses the individual case files and metadata to build a structured knowledge graph
containing judges, parties, citation edges, holdings, statutory references, and
numerical facts. Uses OpenCode CLI for LLM-assisted holding extraction when --extract-holdings
is set.

Usage:
    python data_acquisition/extract_graph.py \
        --corpus-dir corpus_text/legal_casebook \
        --output corpus_text/legal_casebook/metadata/entity_graph.json \
        [--extract-holdings] [--concurrency 8] [--model anthropic/claude-sonnet-4-5]

Requires: tqdm
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import tempfile
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

# Common US statute patterns
STATUTE_PATTERNS = [
    # US Code: "26 USC § 1031", "42 U.S.C. § 1983", "42 U. S. C. § 1983"
    r"\d+\s+U\.?\s*S\.?\s*C\.?\s*§?\s*\d+[a-zA-Z]*(?:\([a-z0-9]+\))*",
    # Named statutes with section numbers
    r"(?:Clean Air Act|Clean Water Act|CERCLA|NEPA|Title VII|ADA|FMLA|"
    r"Lanham Act|DMCA|Sherman Act|Clayton Act|RICO|ERISA|"
    r"Sarbanes-Oxley|Dodd-Frank)\s*(?:§\s*\d+[a-zA-Z]*)?",
    # State codes: "Cal. Civ. Code § 1234", "Tex. Bus. & Com. Code § 17.46"
    r"(?:Cal\.|N\.Y\.|Tex\.)\s+(?:[A-Za-z.&]+\s+)*(?:Code|Law)\s*§\s*\d+[a-zA-Z.]*",
    # CFR: "40 CFR § 261.3"
    r"\d+\s+C\.F\.R\.?\s*§?\s*\d+(?:\.\d+)*",
]

# Patterns for extracting dollar amounts and other numerical facts
NUMERICAL_PATTERNS = {
    "damages": r"\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand)?",
    "statutory_threshold": r"(?:threshold|limit|cap|maximum|minimum)\s+(?:of\s+)?\$[\d,]+(?:\.\d+)?",
    "vote_count": r"(\d+)\s*(?:to|[-–])\s*(\d+)\s+(?:decision|vote|majority|dissent)",
    "time_period": r"(\d+)\s*(?:year|month|day)s?\s+(?:sentence|period|term|statute of limitations)",
}

# Party role indicators
PARTY_INDICATORS = {
    "plaintiff": ["plaintiff", "petitioner", "appellant", "complainant", "relator"],
    "defendant": ["defendant", "respondent", "appellee"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _atomic_save(data: Any, path: str) -> None:
    """Write JSON to .tmp then rename atomically."""
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)


def _parse_case_header(text: str) -> dict[str, str]:
    """Parse structured header fields from a case file's text."""
    header = {}
    # Find header block (everything before the --- delimiter)
    parts = text.split("---", 1)
    header_text = parts[0] if parts else text[:2000]

    for line in header_text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key in ("CASE_ID", "CASE_NAME", "COURT", "DATE", "CITATIONS",
                        "JUDGES", "TOPIC", "JURISDICTION"):
                header[key.lower()] = value
    return header


# ---------------------------------------------------------------------------
# Entity extraction (rule-based)
# ---------------------------------------------------------------------------


def extract_judges(cases: list[dict]) -> list[dict]:
    """Extract unique judges and their associated cases."""
    judge_map: dict[str, dict] = {}  # name -> {court, cases}

    for case in cases:
        case_id = case.get("case_id", "")
        court = case.get("court", "")
        for judge_name in case.get("judges", []):
            if judge_name not in judge_map:
                judge_map[judge_name] = {
                    "name": judge_name,
                    "courts": set(),
                    "cases": [],
                }
            judge_map[judge_name]["courts"].add(court)
            if case_id not in judge_map[judge_name]["cases"]:
                judge_map[judge_name]["cases"].append(case_id)

    result = []
    for name, info in sorted(judge_map.items()):
        result.append({
            "name": name,
            "courts": sorted(info["courts"]),
            "cases": info["cases"],
            "case_count": len(info["cases"]),
        })
    return result


def extract_parties(case_text: str, case_id: str) -> list[dict]:
    """Extract party names and roles from case text."""
    parties = []
    seen = set()

    # Parse "X v. Y" from case name patterns
    vs_pattern = r"([A-Z][A-Za-z.\s&,']+?)\s+v\.\s+([A-Z][A-Za-z.\s&,']+?)(?:\s*[,;(\[]|$)"
    for match in re.finditer(vs_pattern, case_text[:1000]):
        plaintiff = match.group(1).strip().rstrip(",;")
        defendant = match.group(2).strip().rstrip(",;")

        for name, role in [(plaintiff, "plaintiff"), (defendant, "defendant")]:
            if name and name not in seen and len(name) > 2:
                parties.append({
                    "name": name,
                    "type": role,
                    "cases": [case_id],
                })
                seen.add(name)

    # Search for explicit role mentions
    # Common verbs/phrases that should not be captured as party names
    _PARTY_STOP_WORDS = {"filed", "moved", "argued", "appeals", "contends",
                         "appealed", "asserts", "claims", "alleged", "denied",
                         "the", "court", "united", "states"}
    for role, indicators in PARTY_INDICATORS.items():
        for indicator in indicators:
            pattern = rf"\b{indicator}\s+((?:[A-Z][A-Za-z.&'-]+)(?:\s+(?:of|and|the|for|in)\s+(?:[A-Z][A-Za-z.&'-]+))*(?:\s+[A-Z][A-Za-z.&'-]+)*)(?:\s*[,;(]|\s+(?:filed|moved|argued|appeals|contends))"
            for match in re.finditer(pattern, case_text[:5000]):
                name = match.group(1).strip().rstrip(",;")
                words = name.split()
                # Reject captures with >6 words or that start with common verbs
                if (name and name not in seen and len(name) > 2 and len(name) < 80
                        and len(words) <= 6
                        and words[0].lower() not in _PARTY_STOP_WORDS):
                    parties.append({
                        "name": name,
                        "type": role,
                        "cases": [case_id],
                    })
                    seen.add(name)

    return parties


def extract_citation_edges(cases: list[dict]) -> list[dict]:
    """Build citation graph edges from extracted citations.

    Matches extracted_citations against citation strings in case_index.
    """
    # Build lookup: citation string → case_id
    cite_to_id: dict[str, str] = {}
    for case in cases:
        for cite in case.get("citations", []):
            if cite:
                cite_to_id[cite] = case["case_id"]

    edges = []
    for case in cases:
        from_id = case["case_id"]
        for extracted_cite in case.get("extracted_citations", []):
            # Try exact match first
            to_id = cite_to_id.get(extracted_cite)
            if to_id and to_id != from_id:
                edges.append({
                    "from": from_id,
                    "to": to_id,
                    "citation": extracted_cite,
                })
                continue

            # Try partial match (citation string contained in a known citation)
            for known_cite, kid in cite_to_id.items():
                if kid != from_id and extracted_cite in known_cite:
                    edges.append({
                        "from": from_id,
                        "to": kid,
                        "citation": extracted_cite,
                    })
                    break

    # Deduplicate
    seen_edges = set()
    unique_edges = []
    for edge in edges:
        key = (edge["from"], edge["to"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(edge)

    return unique_edges


def extract_statutes(case_text: str, case_id: str) -> list[dict]:
    """Extract statutory references from case text."""
    statutes = []
    seen = set()

    for pattern in STATUTE_PATTERNS:
        for match in re.finditer(pattern, case_text):
            code = match.group().strip()
            # Normalize whitespace
            code = re.sub(r"\s+", " ", code)
            if code not in seen and len(code) > 5:
                statutes.append({
                    "code": code,
                    "cases": [case_id],
                })
                seen.add(code)

    return statutes


MAGNITUDE_MAP = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}


def extract_numerical_facts(case_text: str, case_id: str) -> list[dict]:
    """Extract numerical facts (damages, thresholds, vote counts) from case text."""
    facts = []

    for fact_type, pattern in NUMERICAL_PATTERNS.items():
        for match in re.finditer(pattern, case_text, re.IGNORECASE):
            value_str = match.group()
            # Try to extract a numeric value
            num_match = re.search(r"\$?([\d,]+(?:\.\d+)?)", value_str)
            if num_match:
                try:
                    value = float(num_match.group(1).replace(",", ""))
                    # Apply magnitude multiplier (e.g. "$4 million" → 4_000_000)
                    for word, multiplier in MAGNITUDE_MAP.items():
                        if word in value_str.lower():
                            value *= multiplier
                            break
                except ValueError:
                    value = None
            else:
                value = None

            # Get surrounding context (±50 chars)
            start = max(0, match.start() - 50)
            end = min(len(case_text), match.end() + 50)
            context = case_text[start:end].replace("\n", " ").strip()

            facts.append({
                "case_id": case_id,
                "type": fact_type,
                "raw_text": value_str,
                "value": value,
                "context": context,
            })

    return facts


# ---------------------------------------------------------------------------
# LLM-assisted holding extraction
# ---------------------------------------------------------------------------


def _parse_opencode_json_output(raw: str) -> str:
    """Parse OpenCode --format json output and extract text content."""
    text_parts = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if event.get("type") == "text":
                text = event.get("part", {}).get("text", "")
                if text:
                    text_parts.append(text)
        except json.JSONDecodeError:
            continue
    return "".join(text_parts)


async def extract_holding_llm(
    case_id: str,
    case_text: str,
    topic: str,
    semaphore: asyncio.Semaphore,
    model: str = "opencode/claude-sonnet-4-5",
    opencode_bin: str = "opencode",
    timeout: float = 120.0,
) -> dict | None:
    """Use an LLM via OpenCode to extract the key legal holding from a case."""
    async with semaphore:
        # Truncate very long opinions for the holding extraction prompt
        truncated = case_text[:15000]
        if len(case_text) > 15000:
            truncated += "\n\n[... opinion truncated for holding extraction ...]"

        prompt = (
            "Extract the key legal holding from this court opinion. "
            "Respond with ONLY a JSON object (no markdown, no explanation):\n"
            '{"holding": "<one-sentence holding>", "topic": "<legal topic>", '
            '"reasoning": "<brief description of court\'s reasoning>"}\n\n'
            f"Topic area: {topic}\n\n"
            f"Opinion text:\n{truncated}"
        )

        cmd = [
            opencode_bin, "run",
            "--format", "json",
            "-m", model,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout extracting holding for {case_id}")
            return None
        except Exception as e:
            logger.warning(f"Error extracting holding for {case_id}: {e}")
            return None

        raw = stdout_bytes.decode(errors="replace").strip()
        reply = _parse_opencode_json_output(raw)

        # Extract JSON from reply
        try:
            # Strip markdown code fences
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip(), flags=re.MULTILINE)
            holding = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            match = re.search(r"\{[\s\S]+\}", reply)
            if match:
                try:
                    holding = json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None

        if "holding" not in holding:
            return None

        return {
            "case_id": case_id,
            "holding": holding.get("holding", ""),
            "topic": holding.get("topic", topic),
            "reasoning": holding.get("reasoning", ""),
        }


async def extract_holdings_batch(
    cases: list[dict],
    case_texts: dict[str, str],
    concurrency: int = 8,
    model: str = "anthropic/claude-sonnet-4-5",
) -> list[dict]:
    """Extract holdings for a batch of cases using OpenCode."""
    semaphore = asyncio.Semaphore(concurrency)
    holdings = []

    tasks = []
    for case in cases:
        case_id = case["case_id"]
        text = case_texts.get(case_id, "")
        if not text:
            continue
        tasks.append(
            extract_holding_llm(
                case_id, text, case.get("topic", ""),
                semaphore, model=model,
            )
        )

    logger.info(f"Extracting holdings for {len(tasks)} cases (concurrency={concurrency})...")

    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            holdings.append(result)

    logger.info(f"Extracted {len(holdings)} holdings")
    return holdings


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_entity_graph(
    case_index: dict,
    case_texts: dict[str, str],
    holdings: list[dict] | None = None,
) -> dict:
    """Build the complete entity graph from case index and texts."""
    cases = case_index.get("cases", [])

    # Extract judges
    logger.info("Extracting judges...")
    judges = extract_judges(cases)
    logger.info(f"  Found {len(judges)} unique judges")

    # Extract parties
    logger.info("Extracting parties...")
    all_parties: dict[str, dict] = {}
    for case in cases:
        case_id = case["case_id"]
        text = case_texts.get(case_id, "")
        if not text:
            continue
        for party in extract_parties(text, case_id):
            key = party["name"]
            if key in all_parties:
                if case_id not in all_parties[key]["cases"]:
                    all_parties[key]["cases"].append(case_id)
            else:
                all_parties[key] = party
    parties = sorted(all_parties.values(), key=lambda p: -len(p["cases"]))
    logger.info(f"  Found {len(parties)} unique parties")

    # Extract citation edges
    logger.info("Extracting citation graph...")
    citation_edges = extract_citation_edges(cases)
    logger.info(f"  Found {len(citation_edges)} citation edges")

    # Extract statutes
    logger.info("Extracting statutory references...")
    all_statutes: dict[str, dict] = {}
    for case in cases:
        case_id = case["case_id"]
        text = case_texts.get(case_id, "")
        if not text:
            continue
        for statute in extract_statutes(text, case_id):
            key = statute["code"]
            if key in all_statutes:
                if case_id not in all_statutes[key]["cases"]:
                    all_statutes[key]["cases"].append(case_id)
            else:
                all_statutes[key] = statute
    statutes = sorted(all_statutes.values(), key=lambda s: -len(s["cases"]))
    logger.info(f"  Found {len(statutes)} unique statutory references")

    # Extract numerical facts
    logger.info("Extracting numerical facts...")
    all_numerical: list[dict] = []
    for case in cases:
        case_id = case["case_id"]
        text = case_texts.get(case_id, "")
        if not text:
            continue
        facts = extract_numerical_facts(text, case_id)
        all_numerical.extend(facts)
    logger.info(f"  Found {len(all_numerical)} numerical facts")

    graph = {
        "judges": judges,
        "parties": parties,
        "citation_edges": citation_edges,
        "holdings": holdings or [],
        "statutes": statutes,
        "numerical_facts": all_numerical,
        "summary": {
            "total_cases": len(cases),
            "total_judges": len(judges),
            "total_parties": len(parties),
            "total_citation_edges": len(citation_edges),
            "total_holdings": len(holdings) if holdings else 0,
            "total_statutes": len(statutes),
            "total_numerical_facts": len(all_numerical),
        },
    }

    return graph


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract entity & citation graph from legal casebook corpus."
    )
    parser.add_argument(
        "--corpus-dir", default="corpus_text/legal_casebook",
        help="Directory containing case files and metadata.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for entity_graph.json. "
             "Defaults to {corpus-dir}/metadata/entity_graph.json.",
    )
    parser.add_argument(
        "--extract-holdings", action="store_true",
        help="Use LLM to extract holdings (requires opencode CLI).",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model", default="opencode/claude-sonnet-4-5")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.corpus_dir, "metadata", "entity_graph.json")

    # Load case index
    index_path = os.path.join(args.corpus_dir, "metadata", "case_index.json")
    if not os.path.exists(index_path):
        sys.exit(f"Case index not found: {index_path}\nRun download_cap.py first.")

    with open(index_path) as f:
        case_index = json.load(f)

    logger.info(f"Loaded case index: {case_index['total_cases']} cases")

    # Load case texts
    logger.info("Loading case texts...")
    case_texts: dict[str, str] = {}
    for case_meta in tqdm(case_index["cases"], desc="Loading case files", unit="case"):
        case_id = case_meta["case_id"]
        file_path = os.path.join(args.corpus_dir, case_meta["file"])
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                case_texts[case_id] = f.read()
        else:
            logger.warning(f"Case file missing: {file_path}")

    logger.info(f"Loaded {len(case_texts)} case texts")

    # LLM-assisted holdings extraction (optional)
    holdings = None
    if args.extract_holdings:
        # Check for existing holdings to support resume
        existing_holdings: list[dict] = []
        if os.path.exists(args.output):
            try:
                with open(args.output) as f:
                    existing_graph = json.load(f)
                existing_holdings = existing_graph.get("holdings", [])
                logger.info(f"Resuming: {len(existing_holdings)} holdings already extracted")
            except (json.JSONDecodeError, KeyError):
                pass

        done_ids = {h["case_id"] for h in existing_holdings}
        remaining_cases = [
            c for c in case_index["cases"]
            if c["case_id"] not in done_ids
        ]

        if remaining_cases:
            new_holdings = asyncio.run(
                extract_holdings_batch(
                    remaining_cases, case_texts,
                    concurrency=args.concurrency, model=args.model,
                )
            )
            holdings = existing_holdings + new_holdings
        else:
            holdings = existing_holdings
            logger.info("All holdings already extracted")

    # Build graph
    graph = build_entity_graph(case_index, case_texts, holdings)

    # Save
    _atomic_save(graph, args.output)

    # Summary
    print("\n" + "=" * 60)
    print("ENTITY GRAPH EXTRACTION COMPLETE")
    print("=" * 60)
    summary = graph["summary"]
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

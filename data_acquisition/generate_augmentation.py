#!/usr/bin/env python3
"""Phase 3: Generate LLM-augmented companion documents for the legal casebook.

Generates three layers of synthetic documents referencing real cases:
  Layer A: Litigation documents (briefs, motions, settlement agreements, court orders)
  Layer B: Legislative materials (statutes, committee reports, floor debates, regulatory guidance)
  Layer C: Secondary sources (law review articles, practice guides, treatise sections)

Uses OpenCode Zen HTTP API for fast async LLM generation with concurrency control,
atomic saves, and resume support.

Usage:
    OPENCODE_API_KEY=sk-... python data_acquisition/generate_augmentation.py \
        --corpus-dir corpus_text/legal_casebook \
        --output-dir corpus_text/legal_casebook/augmented \
        --concurrency 8 --model claude-sonnet-4-5 \
        [--layer A] [--layer B] [--layer C] \
        [--max-cases 1000]

Requires: tqdm, aiohttp
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import aiohttp
except ImportError:
    sys.exit("Missing dependency: uv pip install aiohttp")

try:
    from tqdm import tqdm
    from tqdm.asyncio import tqdm as atqdm
except ImportError:
    sys.exit("Missing dependency: uv pip install tqdm")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENCODE_ZEN_URL = "https://opencode.ai/zen/v1/chat/completions"
CHARS_PER_TOKEN = 4.0
SAVE_INTERVAL = 25  # Atomic save every N documents

# Document types per layer
LAYER_A_TYPES = [
    "plaintiff_brief", "defendant_brief", "motion_summary_judgment",
    "motion_discovery", "settlement_agreement", "court_order",
]
LAYER_B_TYPES = [
    "statutory_text", "committee_report", "floor_debate", "regulatory_guidance",
]
LAYER_C_TYPES = [
    "law_review_article", "practice_guide", "treatise_section",
]

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")


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


def _estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def _load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    path = os.path.join(PROMPT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _render(template: str, **kwargs: str) -> str:
    """Simple template rendering with {{VAR}} substitution."""
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"{{{{{key}}}}}", str(value))
    return result


def _get_related_cases(case_id: str, graph: dict, case_index: dict,
                       max_related: int = 5) -> list[dict]:
    """Get related cases from citation graph."""
    related_ids = set()
    for edge in graph.get("citation_edges", []):
        if edge["from"] == case_id:
            related_ids.add(edge["to"])
        elif edge["to"] == case_id:
            related_ids.add(edge["from"])

    # Lookup case metadata
    id_to_case = {c["case_id"]: c for c in case_index.get("cases", [])}
    related = [id_to_case[rid] for rid in related_ids if rid in id_to_case]
    return related[:max_related]


def _get_case_entities(case_id: str, graph: dict) -> dict:
    """Get entities related to a specific case from the graph."""
    entities: dict[str, Any] = {
        "judges": [],
        "parties": [],
        "statutes": [],
        "numerical_facts": [],
        "holdings": [],
    }
    for judge in graph.get("judges", []):
        if case_id in judge.get("cases", []):
            entities["judges"].append(judge["name"])
    for party in graph.get("parties", []):
        if case_id in party.get("cases", []):
            entities["parties"].append({
                "name": party["name"],
                "type": party.get("type", "unknown"),
            })
    for statute in graph.get("statutes", []):
        if case_id in statute.get("cases", []):
            entities["statutes"].append(statute["code"])
    for fact in graph.get("numerical_facts", []):
        if fact.get("case_id") == case_id:
            entities["numerical_facts"].append(fact)
    for holding in graph.get("holdings", []):
        if holding.get("case_id") == case_id:
            entities["holdings"].append(holding)
    return entities


def _get_statute_cases(statute_code: str, graph: dict, case_index: dict) -> list[dict]:
    """Get all cases that reference a particular statute."""
    id_to_case = {c["case_id"]: c for c in case_index.get("cases", [])}
    for statute in graph.get("statutes", []):
        if statute["code"] == statute_code:
            return [id_to_case[cid] for cid in statute["cases"] if cid in id_to_case]
    return []


# ---------------------------------------------------------------------------
# Post-processing: sanitizer and validator
# ---------------------------------------------------------------------------


# Regex for lines containing 3+ consecutive underscores (signature/fill-in lines)
_RE_UNDERSCORE_LINE = re.compile(r"_{3,}")

# Bracket placeholders — match lines containing [Address], [Phone], etc.
_BRACKET_PLACEHOLDERS = re.compile(
    r"\[(?:Address|Phone|Email|Name|Law Firm|Date|City|State|Fax|"
    r"JUDGE SIGNATURE|Title|Counsel[^\]]*)\]",
    re.IGNORECASE,
)

# Bar number lines
_RE_BAR_NUMBER = re.compile(r"Bar\s+No\.|Bar\s+Number|State\s+Bar", re.IGNORECASE)

# Notarization block opener: "STATE OF ___" followed soon by "COUNTY OF" / "PARISH OF"
_RE_NOTARY_START = re.compile(
    r"^[ \t]*STATE\s+OF\b", re.IGNORECASE,
)
_RE_NOTARY_CONFIRM = re.compile(
    r"COUNTY\s+OF|PARISH\s+OF|Notary\s+Public|sworn\s+and\s+subscribed|"
    r"My\s+commission\s+expires",
    re.IGNORECASE,
)

# Certificate of service header
_RE_CERT_SERVICE = re.compile(
    r"^[ \t]*CERTIFICATE\s+OF\s+SERVICE", re.IGNORECASE,
)

# Signature-block openers (must appear at or near the start of a line)
_RE_SIG_BLOCK = re.compile(
    r"^[ \t]*(?:Respectfully\s+submitted|/s/\s|SO\s+ORDERED\s*[.,]?\s*$)",
    re.IGNORECASE,
)


def sanitize_document(text: str) -> str:
    """Strip synthetic artifacts (signature blocks, placeholders, notary sections, etc.)."""
    lines = text.split("\n")
    out: list[str] = []
    skip_rest = False

    for i, line in enumerate(lines):
        if skip_rest:
            break

        # 1. Certificate of service → drop everything from here to end
        if _RE_CERT_SERVICE.match(line):
            break

        # 2. Notarization block: STATE OF … followed by COUNTY/PARISH within 3 lines
        if _RE_NOTARY_START.match(line):
            lookahead = "\n".join(lines[i : i + 5])
            if _RE_NOTARY_CONFIRM.search(lookahead):
                break

        # 3. Signature block openers → drop to end
        if _RE_SIG_BLOCK.match(line):
            # Keep "SO ORDERED" if it's followed by substantive text (the order
            # disposition), but drop it if it's just a closing line before a
            # judge signature.
            if re.match(r"^[ \t]*SO\s+ORDERED", line, re.IGNORECASE):
                remaining = "\n".join(lines[i + 1 :]).strip()
                # If the remaining text is short boilerplate (< 200 chars) or
                # starts with underscores / a date / signature, treat as tail.
                if len(remaining) < 200 or _RE_UNDERSCORE_LINE.search(
                    remaining.split("\n")[0] if remaining else ""
                ):
                    break
                # Otherwise it's part of the order — keep it
            else:
                break

        # 4. Lines with bracket placeholders → skip line
        if _BRACKET_PLACEHOLDERS.search(line):
            continue

        # 5. Lines containing blank signature underscores
        if _RE_UNDERSCORE_LINE.search(line):
            continue

        # 6. Bar number lines
        if _RE_BAR_NUMBER.search(line):
            continue

        out.append(line)

    # 7. Trim trailing blank lines
    while out and out[-1].strip() == "":
        out.pop()

    return "\n".join(out)


def validate_document(text: str, doc_id: str = "") -> list[str]:
    """Return a list of remaining quality issues found after sanitization."""
    issues: list[str] = []

    if re.search(r"\[[^\]]{1,40}\]", text):
        # Check for actual placeholder-style brackets (not legal citations like [1])
        for m in re.finditer(r"\[([^\]]{1,40})\]", text):
            content = m.group(1)
            # Skip footnote numbers, citation pinpoints, and short numerics
            if re.match(r"^\d+$", content):
                continue
            # Skip common legal citation patterns like [hereinafter ...]
            if content.lower().startswith("hereinafter"):
                continue
            issues.append(f"bracketed placeholder: [{content}]")
            if len(issues) >= 5:
                break

    if re.search(r"_{3,}", text):
        issues.append("contains 3+ consecutive underscores")

    if re.search(r"Bar\s+No", text, re.IGNORECASE):
        issues.append("contains 'Bar No'")

    if re.search(r"Notary\s+Public", text, re.IGNORECASE):
        issues.append("contains 'Notary Public'")

    if issues:
        label = f" ({doc_id})" if doc_id else ""
        logger.warning(f"Document quality issues{label}: {'; '.join(issues)}")

    return issues


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------


async def generate_document(
    prompt: str,
    doc_type: str,
    doc_id: str,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    model: str = "claude-sonnet-4-5",
    api_key: str = "",
    max_tokens: int = 8192,
    timeout: float = 300.0,
    max_retries: int = 3,
) -> dict | None:
    """Generate a single synthetic document via OpenCode Zen HTTP API."""
    async with semaphore:
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        data = None
        for attempt in range(max_retries):
            try:
                async with session.post(
                    OPENCODE_ZEN_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        break
                    body = await resp.text()
                    logger.warning(
                        f"API error for {doc_type} {doc_id} "
                        f"(attempt {attempt+1}/{max_retries}): "
                        f"status={resp.status} body={body[:200]}"
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout for {doc_type} {doc_id} "
                    f"(attempt {attempt+1}/{max_retries})"
                )
            except Exception as e:
                logger.warning(
                    f"Error for {doc_type} {doc_id} "
                    f"(attempt {attempt+1}/{max_retries}): {e}"
                )

            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1) + random.random() * 2
                await asyncio.sleep(wait)

        if data is None:
            logger.error(f"Failed after {max_retries} retries: {doc_type} {doc_id}")
            return None

        # Extract text from OpenAI-compatible response
        try:
            reply = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logger.warning(f"Bad response structure for {doc_type} {doc_id}")
            return None

        if not reply or len(reply) < 100:
            logger.warning(f"Short/empty response for {doc_type} {doc_id}")
            return None

        # Post-process: strip synthetic artifacts, then validate
        reply = sanitize_document(reply)
        validate_document(reply, doc_id)

        return {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "content": reply,
            "token_estimate": _estimate_tokens(reply),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
        }


# ---------------------------------------------------------------------------
# Layer A: Litigation Documents
# ---------------------------------------------------------------------------


def build_litigation_prompt(
    case_meta: dict,
    case_text: str,
    doc_type: str,
    related_cases: list[dict],
    entities: dict,
    template: str,
) -> str:
    """Build the prompt for generating a litigation document."""
    # Format related cases as summaries
    related_summaries = ""
    for rc in related_cases:
        related_summaries += (
            f"- {rc.get('case_name', 'Unknown')} ({rc.get('date', '')})\n"
            f"  Court: {rc.get('court', '')}\n"
            f"  Citations: {'; '.join(rc.get('citations', []))}\n"
        )

    # Format entities
    parties_text = "\n".join(
        f"- {p['name']} ({p['type']})" for p in entities.get("parties", [])
    )
    judges_text = ", ".join(entities.get("judges", []))
    statutes_text = "\n".join(f"- {s}" for s in entities.get("statutes", []))
    numerical_text = "\n".join(
        f"- {f.get('type', '')}: {f.get('raw_text', '')} (context: {f.get('context', '')})"
        for f in entities.get("numerical_facts", [])
    )
    holdings_text = "\n".join(
        f"- {h.get('holding', '')}" for h in entities.get("holdings", [])
    )

    # Truncate case text for the prompt (keep under API limits)
    truncated_text = case_text[:6000]
    if len(case_text) > 6000:
        truncated_text += "\n[... opinion truncated ...]"

    return _render(
        template,
        CASE_NAME=case_meta.get("case_name", "Unknown"),
        CASE_ID=case_meta.get("case_id", ""),
        COURT=case_meta.get("court", ""),
        DATE=case_meta.get("date", ""),
        CITATIONS="; ".join(case_meta.get("citations", [])),
        DOC_TYPE=doc_type,
        CASE_TEXT=truncated_text,
        RELATED_CASES=related_summaries or "None identified",
        PARTIES=parties_text or "Not identified",
        JUDGES=judges_text or "Not identified",
        STATUTES=statutes_text or "None identified",
        NUMERICAL_FACTS=numerical_text or "None identified",
        HOLDINGS=holdings_text or "None identified",
    )


async def generate_layer_a(
    case_index: dict,
    case_texts: dict[str, str],
    graph: dict,
    output_dir: str,
    session: aiohttp.ClientSession,
    api_key: str,
    concurrency: int = 8,
    model: str = "claude-sonnet-4-5",
    max_cases: int = 1000,
    resume_docs: dict[str, dict] | None = None,
) -> list[dict]:
    """Generate litigation documents (Layer A)."""
    template = _load_prompt("brief_generator.txt")
    semaphore = asyncio.Semaphore(concurrency)

    cases = case_index.get("cases", [])[:max_cases]
    done_ids = set(resume_docs.keys()) if resume_docs else set()

    tasks = []
    for case_meta in cases:
        case_id = case_meta["case_id"]
        case_text = case_texts.get(case_id, "")
        if not case_text:
            continue

        related = _get_related_cases(case_id, graph, case_index)
        entities = _get_case_entities(case_id, graph)

        # Generate 2-3 document types per case (randomly selected)
        rng = random.Random(case_id)
        doc_types = rng.sample(LAYER_A_TYPES, min(3, len(LAYER_A_TYPES)))

        for doc_type in doc_types:
            doc_id = f"layerA_{case_id}_{doc_type}"
            if doc_id in done_ids:
                continue

            prompt = build_litigation_prompt(
                case_meta, case_text, doc_type, related, entities, template,
            )
            tasks.append(
                generate_document(
                    prompt, doc_type, doc_id, semaphore, session,
                    model=model, api_key=api_key, timeout=300.0,
                )
            )

    logger.info(f"Layer A: {len(tasks)} documents to generate")
    results = list(resume_docs.values()) if resume_docs else []
    buffer: list[dict] = []
    progress_path = os.path.join(output_dir, "layer_a_progress.json")

    for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Layer A"):
        result = await coro
        if result:
            buffer.append(result)

        if len(buffer) >= SAVE_INTERVAL:
            results.extend(buffer)
            buffer = []
            _atomic_save(results, progress_path)

    if buffer:
        results.extend(buffer)
    _atomic_save(results, progress_path)

    logger.info(f"Layer A complete: {len(results)} documents")
    return results


# ---------------------------------------------------------------------------
# Layer B: Legislative Materials
# ---------------------------------------------------------------------------


def build_legislative_prompt(
    statute_code: str,
    related_cases: list[dict],
    doc_type: str,
    template: str,
) -> str:
    """Build the prompt for generating legislative materials."""
    cases_text = ""
    for rc in related_cases:
        cases_text += (
            f"- {rc.get('case_name', 'Unknown')} ({rc.get('date', '')})\n"
            f"  Court: {rc.get('court', '')}\n"
            f"  Citations: {'; '.join(rc.get('citations', []))}\n"
        )

    return _render(
        template,
        STATUTE_CODE=statute_code,
        DOC_TYPE=doc_type,
        RELATED_CASES=cases_text or "None identified",
        NUM_CASES=str(len(related_cases)),
    )


async def generate_layer_b(
    case_index: dict,
    graph: dict,
    output_dir: str,
    session: aiohttp.ClientSession,
    api_key: str,
    concurrency: int = 8,
    model: str = "claude-sonnet-4-5",
    resume_docs: dict[str, dict] | None = None,
) -> list[dict]:
    """Generate legislative materials (Layer B)."""
    template = _load_prompt("legislative_generator.txt")
    semaphore = asyncio.Semaphore(concurrency)

    # Get statutes referenced by multiple cases
    statutes = [s for s in graph.get("statutes", []) if len(s.get("cases", [])) >= 2]
    statutes.sort(key=lambda s: -len(s["cases"]))
    statutes = statutes[:200]  # Cap at top 200 statutes

    done_ids = set(resume_docs.keys()) if resume_docs else set()
    tasks = []

    for statute in statutes:
        statute_code = statute["code"]
        related = _get_statute_cases(statute_code, graph, case_index)

        # Generate 2-3 document types per statute
        rng = random.Random(statute_code)
        doc_types = rng.sample(LAYER_B_TYPES, min(3, len(LAYER_B_TYPES)))

        for doc_type in doc_types:
            doc_id = f"layerB_{re.sub(r'[^a-zA-Z0-9]', '_', statute_code)}_{doc_type}"
            if doc_id in done_ids:
                continue

            prompt = build_legislative_prompt(
                statute_code, related, doc_type, template,
            )
            tasks.append(
                generate_document(
                    prompt, doc_type, doc_id, semaphore, session,
                    model=model, api_key=api_key, timeout=300.0,
                )
            )

    logger.info(f"Layer B: {len(tasks)} documents to generate")
    results = list(resume_docs.values()) if resume_docs else []
    buffer: list[dict] = []
    progress_path = os.path.join(output_dir, "layer_b_progress.json")

    for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Layer B"):
        result = await coro
        if result:
            buffer.append(result)

        if len(buffer) >= SAVE_INTERVAL:
            results.extend(buffer)
            buffer = []
            _atomic_save(results, progress_path)

    if buffer:
        results.extend(buffer)
    _atomic_save(results, progress_path)

    logger.info(f"Layer B complete: {len(results)} documents")
    return results


# ---------------------------------------------------------------------------
# Layer C: Secondary Sources
# ---------------------------------------------------------------------------


def build_secondary_prompt(
    topic: str,
    cases_in_topic: list[dict],
    doc_type: str,
    graph: dict,
    template: str,
) -> str:
    """Build the prompt for generating secondary source materials."""
    # Format cases for this topic
    cases_text = ""
    for case in cases_in_topic[:20]:
        cases_text += (
            f"- {case.get('case_name', 'Unknown')} ({case.get('date', '')})\n"
            f"  Court: {case.get('court', '')}, Jurisdiction: {case.get('jurisdiction', '')}\n"
        )

    # Find relevant statutes for the topic
    topic_case_ids = {c["case_id"] for c in cases_in_topic}
    relevant_statutes = [
        s["code"] for s in graph.get("statutes", [])
        if any(cid in topic_case_ids for cid in s.get("cases", []))
    ][:15]

    # Find relevant holdings for the topic
    relevant_holdings = [
        h for h in graph.get("holdings", [])
        if h.get("case_id") in topic_case_ids
    ][:10]
    holdings_text = "\n".join(
        f"- [{h.get('case_id', '')}] {h.get('holding', '')}"
        for h in relevant_holdings
    )

    return _render(
        template,
        TOPIC=topic,
        DOC_TYPE=doc_type,
        CASES=cases_text or "None available",
        NUM_CASES=str(len(cases_in_topic)),
        STATUTES="\n".join(f"- {s}" for s in relevant_statutes) or "None identified",
        HOLDINGS=holdings_text or "None available",
    )


async def generate_layer_c(
    case_index: dict,
    graph: dict,
    output_dir: str,
    session: aiohttp.ClientSession,
    api_key: str,
    concurrency: int = 8,
    model: str = "claude-sonnet-4-5",
    resume_docs: dict[str, dict] | None = None,
) -> list[dict]:
    """Generate secondary source materials (Layer C)."""
    template = _load_prompt("secondary_source_generator.txt")
    semaphore = asyncio.Semaphore(concurrency)

    # Group cases by topic
    topic_groups: dict[str, list[dict]] = {}
    for case in case_index.get("cases", []):
        topic = case.get("topic", "general")
        topic_groups.setdefault(topic, []).append(case)

    done_ids = set(resume_docs.keys()) if resume_docs else set()
    tasks = []

    for topic, topic_cases in topic_groups.items():
        if len(topic_cases) < 5:
            continue

        # Generate multiple articles/guides per topic
        for doc_type in LAYER_C_TYPES:
            # Generate multiple articles exploring different aspects
            num_docs = min(5, max(1, len(topic_cases) // 20))
            for idx in range(num_docs):
                doc_id = f"layerC_{re.sub(r'[^a-zA-Z0-9]', '_', topic)}_{doc_type}_{idx}"
                if doc_id in done_ids:
                    continue

                # Select a subset of cases for each document
                rng = random.Random(doc_id)
                subset = rng.sample(topic_cases, min(20, len(topic_cases)))

                prompt = build_secondary_prompt(
                    topic, subset, doc_type, graph, template,
                )
                tasks.append(
                    generate_document(
                        prompt, doc_type, doc_id, semaphore, session,
                        model=model, api_key=api_key, timeout=300.0,
                    )
                )

    logger.info(f"Layer C: {len(tasks)} documents to generate")
    results = list(resume_docs.values()) if resume_docs else []
    buffer: list[dict] = []
    progress_path = os.path.join(output_dir, "layer_c_progress.json")

    for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Layer C"):
        result = await coro
        if result:
            buffer.append(result)

        if len(buffer) >= SAVE_INTERVAL:
            results.extend(buffer)
            buffer = []
            _atomic_save(results, progress_path)

    if buffer:
        results.extend(buffer)
    _atomic_save(results, progress_path)

    logger.info(f"Layer C complete: {len(results)} documents")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_resume_state(output_dir: str, layer: str) -> dict[str, dict]:
    """Load previously generated documents for resume support."""
    progress_path = os.path.join(output_dir, f"layer_{layer.lower()}_progress.json")
    if not os.path.exists(progress_path):
        return {}
    try:
        with open(progress_path) as f:
            docs = json.load(f)
        return {d["doc_id"]: d for d in docs}
    except (json.JSONDecodeError, KeyError):
        return {}


async def run_pipeline(args):
    """Run the generation pipeline with a shared aiohttp session."""
    api_key = os.environ.get("OPENCODE_API_KEY", "")
    if not api_key:
        sys.exit("OPENCODE_API_KEY env var not set.")

    layers = args.layer or ["A", "B", "C"]

    # Load case index
    index_path = os.path.join(args.corpus_dir, "metadata", "case_index.json")
    if not os.path.exists(index_path):
        sys.exit(f"Case index not found: {index_path}\nRun download_cap.py first.")
    with open(index_path) as f:
        case_index = json.load(f)

    # Load entity graph
    graph_path = os.path.join(args.corpus_dir, "metadata", "entity_graph.json")
    if not os.path.exists(graph_path):
        sys.exit(f"Entity graph not found: {graph_path}\nRun extract_graph.py first.")
    with open(graph_path) as f:
        graph = json.load(f)

    logger.info(f"Loaded {case_index['total_cases']} cases, "
                f"{graph['summary']['total_citation_edges']} citation edges")

    # Load case texts (needed for Layer A)
    case_texts: dict[str, str] = {}
    if "A" in layers:
        logger.info("Loading case texts for Layer A...")
        for case_meta in tqdm(case_index["cases"], desc="Loading cases", unit="case"):
            file_path = os.path.join(args.corpus_dir, case_meta["file"])
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    case_texts[case_meta["case_id"]] = f.read()

    all_results: dict[str, list[dict]] = {}
    total_tokens = 0

    # Single shared HTTP session for all requests
    async with aiohttp.ClientSession() as session:
        if "A" in layers:
            resume_a = load_resume_state(args.output_dir, "a") if args.resume else {}
            results_a = await generate_layer_a(
                case_index, case_texts, graph, args.output_dir,
                session, api_key,
                concurrency=args.concurrency, model=args.model,
                max_cases=args.max_cases, resume_docs=resume_a,
            )
            all_results["A"] = results_a
            total_tokens += sum(d.get("token_estimate", 0) for d in results_a)

        if "B" in layers:
            resume_b = load_resume_state(args.output_dir, "b") if args.resume else {}
            results_b = await generate_layer_b(
                case_index, graph, args.output_dir,
                session, api_key,
                concurrency=args.concurrency, model=args.model,
                resume_docs=resume_b,
            )
            all_results["B"] = results_b
            total_tokens += sum(d.get("token_estimate", 0) for d in results_b)

        if "C" in layers:
            resume_c = load_resume_state(args.output_dir, "c") if args.resume else {}
            results_c = await generate_layer_c(
                case_index, graph, args.output_dir,
                session, api_key,
                concurrency=args.concurrency, model=args.model,
                resume_docs=resume_c,
            )
            all_results["C"] = results_c
            total_tokens += sum(d.get("token_estimate", 0) for d in results_c)

    return all_results, total_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM-augmented companion documents for legal casebook."
    )
    parser.add_argument(
        "--corpus-dir", default="corpus_text/legal_casebook",
        help="Directory containing case files and metadata.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for augmented documents. "
             "Defaults to {corpus-dir}/augmented/.",
    )
    parser.add_argument(
        "--layer", action="append", choices=["A", "B", "C"],
        help="Which layer(s) to generate. Can be specified multiple times. "
             "Defaults to all layers.",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model", default="claude-sonnet-4-5")
    parser.add_argument("--max-cases", type=int, default=1000,
                        help="Max cases for Layer A generation.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing progress files.")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.corpus_dir, "augmented")
    os.makedirs(args.output_dir, exist_ok=True)

    all_results, total_tokens = asyncio.run(run_pipeline(args))

    # Write individual document files
    logger.info("Writing individual document files...")
    for layer, docs in all_results.items():
        layer_dir = os.path.join(args.output_dir, f"layer_{layer.lower()}")
        os.makedirs(layer_dir, exist_ok=True)
        for doc in docs:
            doc_path = os.path.join(layer_dir, f"{doc['doc_id']}.txt")
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(doc["content"])

    # Summary
    print("\n" + "=" * 60)
    print("AUGMENTATION COMPLETE")
    print("=" * 60)
    for layer, docs in all_results.items():
        layer_tokens = sum(d.get("token_estimate", 0) for d in docs)
        print(f"  Layer {layer}: {len(docs)} documents, ~{layer_tokens:,} tokens")
    print(f"  Total: ~{total_tokens:,} tokens")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

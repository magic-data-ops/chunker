#!/usr/bin/env python3
"""Contractor LLM: validate and polish each Q&A chain.

Usage:
    python 3_contractor_polish.py --input qa_chains_raw.json --output qa_chains_validated.json

Mirrors the async + resume + atomic-save pattern from evaluation/6_generate_rubrics.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as atqdm

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from utils.llm_client import QAAgent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
CATEGORIES_CFG = os.path.join(os.path.dirname(__file__), "qa_config", "categories.yaml")
CONCURRENCY_LIMIT = 32
SAVE_INTERVAL = 50


def _load_categories() -> dict:
    with open(CATEGORIES_CFG, "r") as f:
        cfg = yaml.safe_load(f)
    return {c["name"]: c["description"] for c in cfg["categories"]}


def _load_prompt(name: str) -> str:
    with open(os.path.join(PROMPT_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def _render(template: str, **kwargs) -> str:
    for key, value in kwargs.items():
        template = template.replace("{{" + key + "}}", str(value) if value is not None else "")
    return template


# ---------------------------------------------------------------------------
# Hop path text formatter
# ---------------------------------------------------------------------------


def _format_hop_path(hop_path: List[dict]) -> str:
    lines = []
    for hop in hop_path:
        score_str = f" (retrieval_score={hop['retrieval_score']:.3f})" if hop["retrieval_score"] else ""
        lines.append(
            f"[Hop {hop['hop_index']}] chunk_id={hop['chunk_id']}{score_str}\n"
            f"  Chunk text (excerpt): {hop['chunk_text'][:300]}{'…' if len(hop['chunk_text']) > 300 else ''}\n"
            f"  Partial answer: {hop['partial_answer']}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Contractor validation
# ---------------------------------------------------------------------------


async def validate_chain(chain: dict, agent: QAAgent, validator_tpl: str,
                         category_descriptions: dict, semaphore: asyncio.Semaphore) -> Optional[dict]:
    async with semaphore:
        cat_name = chain.get("category", "unknown")
        cat_desc = category_descriptions.get(cat_name, "")
        hop_path_text = _format_hop_path(chain.get("hop_path", []))

        prompt = _render(
            validator_tpl,
            CHAIN_ID=chain["chain_id"],
            CATEGORY=cat_name,
            CATEGORY_DESCRIPTION=cat_desc,
            HOP_COUNT=str(chain.get("hop_count", 0)),
            TERMINATION_REASON=chain.get("termination_reason", "unknown"),
            QUESTION=chain.get("question", ""),
            HOP_PATH_TEXT=hop_path_text,
        )

        raw = await agent.generate(prompt)

        # Parse JSON response
        validation: dict = {}
        try:
            # Strip markdown code fences if present
            json_str = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
            validation = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            # Try extracting first {...} block
            match = re.search(r"\{[\s\S]+\}", raw)
            if match:
                try:
                    validation = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if not validation:
            print(f"  [WARN] Failed to parse contractor response for chain {chain['chain_id']}")
            validation = {
                "approved": False,
                "category_suitability_score": 0.0,
                "answer_completeness_score": 0.0,
                "polished_answer": chain.get("final_answer", ""),
                "rejection_reason": "contractor_parse_error",
            }

        result = chain.copy()
        result.update(validation)
        result["validated_at"] = datetime.now(timezone.utc).isoformat()
        return result


# ---------------------------------------------------------------------------
# Atomic save
# ---------------------------------------------------------------------------


def _atomic_save(data: list, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Contractor LLM: validate + polish Q&A chains")
    parser.add_argument("--input", default="qa_chains_raw.json", help="Raw chains from step 2")
    parser.add_argument("--output", default="qa_chains_validated.json")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY_LIMIT)
    parser.add_argument("--categories_cfg", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        raw_chains: List[dict] = json.load(f)
    print(f"Loaded {len(raw_chains)} chains from {args.input}.")

    # Resume support
    validated: List[dict] = []
    done_ids: set = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            try:
                validated = json.load(f)
                done_ids = {c["chain_id"] for c in validated}
                print(f"Resuming — {len(validated)} already validated.")
            except json.JSONDecodeError:
                print("Output file invalid/empty. Starting fresh.")

    to_process = [c for c in raw_chains if c["chain_id"] not in done_ids]
    print(f"{len(to_process)} chains remaining.")

    if not to_process:
        print("Nothing to do.")
        return

    cat_cfg_path = args.categories_cfg or CATEGORIES_CFG
    if os.path.exists(cat_cfg_path):
        category_descriptions = _load_categories()
    else:
        category_descriptions = {}

    validator_tpl = _load_prompt("contractor_validator.txt")
    agent = QAAgent("contractor_validator")
    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = [validate_chain(c, agent, validator_tpl, category_descriptions, semaphore)
             for c in to_process]

    buffer: List[dict] = []
    pbar = atqdm(total=len(tasks), desc="Validating chains")

    for future in asyncio.as_completed(tasks):
        result = await future
        if result:
            buffer.append(result)
        pbar.update(1)

        if len(buffer) >= SAVE_INTERVAL:
            validated.extend(buffer)
            buffer = []
            _atomic_save(validated, args.output)

    pbar.close()

    if buffer:
        validated.extend(buffer)

    _atomic_save(validated, args.output)
    print(f"Done. {len(validated)} validated chains saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

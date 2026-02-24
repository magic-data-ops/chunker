#!/usr/bin/env python3
"""Generate multi-turn conversation history for existing Q&A chains.

Takes single-turn chains (from step 2) and enriches each with a conversation
history that sets up the evaluation scenario. Uses an LLM to generate natural
conversation turns aligned with each chain's category.

Usage:
    python generate_multiturn.py --input qa_chains_raw.json --output qa_chains_multiturn.json
    python generate_multiturn.py --input qa_chains_raw.json --output qa_chains_multiturn.json --model opus
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import List, Optional, Tuple

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
CONCURRENCY_LIMIT = 16
SAVE_INTERVAL = 20


def _load_categories(cfg_path: str) -> dict:
    """Load categories indexed by name, with full config."""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return {c["name"]: c for c in cfg["categories"]}


def _load_prompt(name: str) -> str:
    with open(os.path.join(PROMPT_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def _render(template: str, **kwargs) -> str:
    for key, value in kwargs.items():
        template = template.replace("{{" + key + "}}", str(value) if value is not None else "")
    return template


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_evidence(chain: dict) -> str:
    """Build a human-readable evidence block from hop_path."""
    parts = []
    for hop in chain.get("hop_path", []):
        text = hop.get("chunk_text", "")
        if text:
            parts.append(f"[Evidence {hop.get('hop_index', '?')}]: {text}")
    if not parts:
        # Fallback to top-level evidence_snippets
        for i, snip in enumerate(chain.get("evidence_snippets", [])):
            if snip:
                parts.append(f"[Evidence {i}]: {snip}")
    return "\n\n".join(parts) if parts else "(no evidence available)"


def _format_entities(chain: dict) -> str:
    """Build a human-readable entities block."""
    entities = chain.get("entities", [])
    if not entities:
        return "(no entities)"
    parts = []
    for ent in entities:
        parts.append(f"- {ent.get('label', '?')}: {ent.get('description', '')}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def _parse_conversation_history(raw: str) -> Optional[list]:
    """Parse the LLM's JSON response into a list of turn dicts."""
    raw = raw.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Try direct parse
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "conversation_history" in data:
            return data["conversation_history"]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting first JSON array
    match = re.search(r"\[[\s\S]*\]", raw)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _validate_conversation_history(
    turns: list,
    expected_count: int,
) -> Tuple[bool, str]:
    """Validate that conversation_history meets schema requirements."""
    if not isinstance(turns, list):
        return False, "conversation_history is not a list"
    if len(turns) != expected_count:
        return False, f"expected {expected_count} turns, got {len(turns)}"

    for i, turn in enumerate(turns):
        if not isinstance(turn, dict):
            return False, f"turn {i} is not a dict"
        if "user" not in turn or "assistant" not in turn:
            return False, f"turn {i} missing 'user' or 'assistant'"
        if not isinstance(turn.get("user"), str) or not turn["user"].strip():
            return False, f"turn {i} has empty 'user'"
        if not isinstance(turn.get("assistant"), str) or not turn["assistant"].strip():
            return False, f"turn {i} has empty 'assistant'"
        if len(turn["user"].strip()) < 10:
            return False, f"turn {i} user message too short ({len(turn['user'].strip())} chars)"
        if len(turn["assistant"].strip()) < 20:
            return False, f"turn {i} assistant message too short ({len(turn['assistant'].strip())} chars)"

    return True, "ok"


# ---------------------------------------------------------------------------
# Per-chain generation
# ---------------------------------------------------------------------------


async def generate_conversation_history(
    chain: dict,
    agent: QAAgent,
    multiturn_tpl: str,
    category_config: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 2,
) -> Tuple[Optional[dict], dict]:
    """Generate multi-turn conversation history for a single chain.

    Returns (enriched_chain_or_None, log_entry).
    """
    chain_id = chain["chain_id"]
    category = chain.get("category", "")
    log = {
        "chain_id": chain_id,
        "category": category,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_turns": category_config.get("num_turns", 1),
        "attempts": [],
        "status": "pending",
    }

    async with semaphore:
        num_turns = category_config.get("num_turns", 1)

        # Single-turn: passthrough with empty history
        if num_turns <= 1:
            result = chain.copy()
            result["num_turns"] = 1
            result["conversation_history"] = []
            result["multiturn_generated_at"] = datetime.now(timezone.utc).isoformat()
            log["status"] = "passthrough"
            return result, log

        history_turns = num_turns - 1
        multiturn_scenario = category_config.get("multiturn_scenario", "")
        evidence_text = _format_evidence(chain)
        entities_text = _format_entities(chain)

        prompt = _render(
            multiturn_tpl,
            CATEGORY_NAME=category_config.get("name", ""),
            CATEGORY_DESCRIPTION=category_config.get("description", "").strip(),
            MULTITURN_SCENARIO=multiturn_scenario.strip() if multiturn_scenario else "",
            NUM_HISTORY_TURNS=str(history_turns),
            NUM_TOTAL_TURNS=str(num_turns),
            QUESTION=chain.get("question", ""),
            GOLDEN_ANSWER=chain.get("final_answer", ""),
            EVIDENCE_TEXT=evidence_text,
            ENTITIES_TEXT=entities_text,
            DISAMBIGUATION_STATEMENT=chain.get("disambiguation_statement", ""),
            DIFFICULTY=chain.get("difficulty", "medium"),
        )

        for attempt in range(max_retries + 1):
            attempt_log: dict = {"attempt": attempt + 1}
            raw = await agent.generate(prompt)

            if raw.startswith("[ERROR]"):
                attempt_log["error"] = raw[:500]
                log["attempts"].append(attempt_log)
                if attempt < max_retries:
                    continue
                print(f"  [WARN] LLM error for chain {chain_id}: {raw[:200]}")
                log["status"] = "failed_llm_error"
                return None, log

            attempt_log["raw_response_preview"] = raw[:500]
            attempt_log["raw_response_length"] = len(raw)

            turns = _parse_conversation_history(raw)

            if turns is None:
                attempt_log["parse_error"] = "could not extract JSON array"
                log["attempts"].append(attempt_log)
                if attempt < max_retries:
                    continue
                print(f"  [WARN] Failed to parse multiturn for chain {chain_id}")
                log["status"] = "failed_parse"
                return None, log

            ok, reason = _validate_conversation_history(turns, history_turns)
            if ok:
                # Normalize turn_index only after validation confirms all turns are dicts
                for i, turn in enumerate(turns):
                    turn["turn_index"] = i + 1
                result = chain.copy()
                result["num_turns"] = num_turns
                result["conversation_history"] = turns
                result["multiturn_generated_at"] = datetime.now(timezone.utc).isoformat()
                attempt_log["validation"] = "ok"
                attempt_log["turns_generated"] = len(turns)
                log["attempts"].append(attempt_log)
                log["status"] = "success"
                log["total_attempts"] = attempt + 1
                return result, log

            attempt_log["validation_error"] = reason
            log["attempts"].append(attempt_log)
            if attempt < max_retries:
                continue
            print(f"  [WARN] Invalid multiturn for chain {chain_id}: {reason}")
            log["status"] = "failed_validation"
            return None, log


# ---------------------------------------------------------------------------
# Atomic save
# ---------------------------------------------------------------------------


def _atomic_save(data: list, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)


# ---------------------------------------------------------------------------
# Run logging
# ---------------------------------------------------------------------------


def _save_run_log(logs: List[dict], log_dir: str) -> None:
    """Save structured JSON logs for the multi-turn generation run."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(log_dir, f"multiturn_{ts}.json")

    summary = {
        "timestamp": ts,
        "total": len(logs),
        "success": sum(1 for l in logs if l["status"] == "success"),
        "passthrough": sum(1 for l in logs if l["status"] == "passthrough"),
        "failed": sum(1 for l in logs if l["status"].startswith("failed")),
        "chains": logs,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Run log saved to {log_path}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def _export_csv(enriched: List[dict], csv_path: str) -> None:
    """Export multi-turn enriched chains to CSV.

    Columns: chain_id, category, num_turns, question, final_answer,
             conversation_history, evidence_snippets
    """
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = csv_path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "chain_id",
            "category",
            "num_turns",
            "question",
            "final_answer",
            "conversation_history",
            "evidence_snippets",
        ])
        for chain in enriched:
            evidence = [
                hop.get("chunk_text", "")
                for hop in chain.get("hop_path", [])
                if hop.get("chunk_text")
            ]
            if not evidence:
                evidence = [s for s in chain.get("evidence_snippets", []) if s]
            writer.writerow([
                chain.get("chain_id", ""),
                chain.get("category", ""),
                chain.get("num_turns", 1),
                chain.get("question", ""),
                chain.get("final_answer", ""),
                json.dumps(chain.get("conversation_history", []), indent=2),
                json.dumps(evidence, indent=2),
            ])
    if os.path.exists(csv_path):
        os.remove(csv_path)
    os.rename(tmp, csv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-turn conversation history for Q&A chains"
    )
    parser.add_argument("--input", required=True,
                        help="Input chains JSON (raw or validated)")
    parser.add_argument("--output", default="qa_chains_multiturn.json",
                        help="Output path for enriched chains")
    parser.add_argument("--csv-output", default=None,
                        help="Output path for CSV export (default: derived from --output)")
    parser.add_argument("--log-dir", default="logs",
                        help="Directory for structured run logs (default: logs)")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY_LIMIT,
                        help=f"Max concurrent LLM calls (default: {CONCURRENCY_LIMIT})")
    parser.add_argument("--categories_cfg", default=None,
                        help="Path to categories YAML config")
    parser.add_argument("--model", default="sonnet",
                        choices=["sonnet", "opus", "haiku"],
                        help="Model for conversation generation (default: sonnet)")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max retries per chain on parse/validation failure")
    parser.add_argument("--max-budget-usd", type=float, default=0.10,
                        help="Cost cap per LLM invocation in USD (default: 0.10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of chains to process (for quick testing)")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer (>= 1)")

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        raw_chains: List[dict] = json.load(f)
    print(f"Loaded {len(raw_chains)} chains from {args.input}.")

    # Load categories
    cat_cfg_path = args.categories_cfg or CATEGORIES_CFG
    categories = _load_categories(cat_cfg_path)
    print(f"Loaded {len(categories)} categories from {cat_cfg_path}.")

    # Resume support
    enriched: List[dict] = []
    done_ids: set = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            try:
                enriched = json.load(f)
                done_ids = {c["chain_id"] for c in enriched}
                print(f"Resuming — {len(enriched)} already processed.")
            except json.JSONDecodeError:
                print("Output file invalid/empty. Starting fresh.")

    to_process = [c for c in raw_chains if c["chain_id"] not in done_ids]

    # Apply --limit
    if args.limit is not None and len(to_process) > args.limit:
        to_process = to_process[: args.limit]
        print(f"Limiting to {args.limit} chains (--limit).")

    # Separate single-turn (no LLM call) from multi-turn
    single_turn = []
    multi_turn = []
    for chain in to_process:
        cat_name = chain.get("category", "")
        cat_config = categories.get(cat_name, {})
        num_turns = cat_config.get("num_turns", 1)
        if num_turns <= 1:
            single_turn.append(chain)
        else:
            multi_turn.append(chain)

    print(f"{len(to_process)} chains remaining: {len(single_turn)} single-turn (passthrough), "
          f"{len(multi_turn)} multi-turn (LLM generation).")

    run_logs: List[dict] = []

    if not to_process:
        print("Nothing to do — all chains already processed.")
        # Still emit log + CSV for the already-enriched data
        output_dir = os.path.dirname(args.output) or "."
        log_dir = args.log_dir
        if not os.path.isabs(log_dir):
            log_dir = os.path.join(output_dir, log_dir)
        _save_run_log(run_logs, log_dir)
        csv_path = args.csv_output or os.path.splitext(args.output)[0] + ".csv"
        _export_csv(enriched, csv_path)
        print(f"CSV: {len(enriched)} rows -> {csv_path}")
        return

    # Process single-turn chains immediately (no LLM call)
    for chain in single_turn:
        result = chain.copy()
        result["num_turns"] = 1
        result["conversation_history"] = []
        result["multiturn_generated_at"] = datetime.now(timezone.utc).isoformat()
        enriched.append(result)
        run_logs.append({
            "chain_id": chain["chain_id"],
            "category": chain.get("category", ""),
            "timestamp": result["multiturn_generated_at"],
            "num_turns": 1,
            "attempts": [],
            "status": "passthrough",
        })

    if single_turn:
        _atomic_save(enriched, args.output)
        print(f"  {len(single_turn)} single-turn chains passed through.")

    if not multi_turn:
        print("No multi-turn chains to process.")
        _atomic_save(enriched, args.output)
    else:
        # Generate multi-turn conversation history
        multiturn_tpl = _load_prompt("multiturn_generator.txt")
        agent = QAAgent("multiturn_generator", model=args.model,
                        max_budget_usd=args.max_budget_usd)
        semaphore = asyncio.Semaphore(args.concurrency)

        tasks = []
        for chain in multi_turn:
            cat_name = chain.get("category", "")
            cat_config = categories.get(cat_name, {})
            tasks.append(
                generate_conversation_history(
                    chain, agent, multiturn_tpl, cat_config,
                    semaphore, max_retries=args.max_retries,
                )
            )

        buffer: List[dict] = []
        skipped = 0
        pbar = atqdm(total=len(tasks), desc="Generating multi-turn")

        for future in asyncio.as_completed(tasks):
            result, log_entry = await future
            run_logs.append(log_entry)
            if result:
                buffer.append(result)
            else:
                skipped += 1
            pbar.update(1)

            if len(buffer) >= SAVE_INTERVAL:
                enriched.extend(buffer)
                buffer = []
                _atomic_save(enriched, args.output)

        pbar.close()

        if buffer:
            enriched.extend(buffer)

        _atomic_save(enriched, args.output)

    total_multiturn = sum(1 for c in enriched if c.get("num_turns", 1) > 1)
    print(f"Done. {len(enriched)} chains saved to {args.output} "
          f"({total_multiturn} multi-turn, {len(enriched) - total_multiturn} single-turn).")

    # Save run log
    output_dir = os.path.dirname(args.output) or "."
    log_dir = args.log_dir
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(output_dir, log_dir)
    _save_run_log(run_logs, log_dir)

    # Export CSV
    csv_path = args.csv_output
    if not csv_path:
        base = os.path.splitext(args.output)[0]
        csv_path = base + ".csv"
    _export_csv(enriched, csv_path)
    print(f"CSV: {len(enriched)} rows -> {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())

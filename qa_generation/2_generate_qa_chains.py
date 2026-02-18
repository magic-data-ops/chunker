#!/usr/bin/env python3
"""Generate Q&A evaluation chains using `opencode run` as the agent harness.

OpenCode's built-in grep and read tools navigate the text corpus autonomously.
For each category we run `opencode run --agent qa_generator --format json`
from the corpus text directory, parse the JSON event stream, and extract the
structured Q&A pairs.

No server needed — each invocation is a standalone `opencode run` subprocess.

Usage:
    python 2_generate_qa_chains.py --corpus_text_dir ./corpus_text --output qa_chains_raw.json
    python 2_generate_qa_chains.py --corpus_text_dir ./corpus_text --n_chains 200 --batch_size 5
    python 2_generate_qa_chains.py --rank 0 --world_size 4  # sharding (categories level)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.resolve()
PROMPT_DIR = SCRIPT_DIR / "prompts"
CATEGORIES_CFG = SCRIPT_DIR / "qa_config" / "categories.yaml"
AGENT_NAME = "qa_generator"


def _find_opencode() -> str:
    if shutil.which("opencode"):
        return "opencode"
    candidates = [
        Path.home() / ".opencode" / "bin" / "opencode",
        Path.home() / ".local" / "bin" / "opencode",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "opencode"


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def _render(template: str, **kwargs) -> str:
    for key, value in kwargs.items():
        template = template.replace("{{" + key + "}}", str(value) if value is not None else "")
    return template


# ---------------------------------------------------------------------------
# OpenCode run wrapper
# ---------------------------------------------------------------------------


def _parse_opencode_json_events(stdout: str) -> str:
    """Extract assistant text from `opencode run --format json` output."""
    text_parts = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if event.get("type") == "text":
                text_parts.append(event["part"]["text"])
        except (json.JSONDecodeError, KeyError):
            continue
    return "\n".join(text_parts)


async def _run_opencode(
    prompt: str,
    corpus_text_dir: str,
    opencode_bin: str,
    timeout: float = 600.0,
) -> str:
    """Run `opencode run` and return the assistant's text response."""
    proc = await asyncio.create_subprocess_exec(
        opencode_bin, "run",
        "--agent", AGENT_NAME,
        "--format", "json",
        "--dir", corpus_text_dir,
        prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return ""
    return _parse_opencode_json_events(stdout.decode(errors="replace"))


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def _extract_json_array(text: str) -> Optional[List[dict]]:
    """Try to extract and parse the first JSON array from an LLM response."""
    text = text.strip()

    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass

    # Find first [...] block
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Chain conversion
# ---------------------------------------------------------------------------


def _pairs_to_chains(pairs: List[dict], category: dict, seed_file: str) -> List[dict]:
    """Convert the agent's raw JSON pairs into the canonical chain schema."""
    chains = []
    for pair in pairs:
        q = pair.get("question", "").strip()
        a = pair.get("golden_answer", "").strip()
        if not q or not a:
            continue
        evidence = pair.get("evidence_snippets", [])
        sources = pair.get("source_files", [seed_file])

        hop_path = [
            {
                "hop_index": i,
                "chunk_id": f"{src}:evidence_{i}",
                "chunk_text": snip,
                "partial_answer": snip,
                "retrieval_score": None,
            }
            for i, (snip, src) in enumerate(
                zip(evidence, sources + [sources[-1]] * max(0, len(evidence) - len(sources)))
            )
        ]

        chains.append({
            "chain_id": str(uuid.uuid4()),
            "category": category["name"],
            "seed_chunk_id": sources[0] if sources else "unknown",
            "question": q,
            "final_answer": a,
            "hop_path": hop_path,
            "hop_count": max(1, len(hop_path)),
            "termination_reason": "agent_complete",
            "single_answer_heuristic": category.get("max_hops", 2) == 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
    return chains


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
# Per-category generation
# ---------------------------------------------------------------------------


async def generate_for_category(
    category: dict,
    corpus_text_dir: str,
    n_chains: int,
    opencode_bin: str,
    batch_size: int,
    gen_template: str,
) -> List[dict]:
    """Generate n_chains Q&A pairs for one category via `opencode run`."""
    txt_files = sorted(Path(corpus_text_dir).glob("*.txt"))
    if not txt_files:
        print(f"  [WARN] No .txt files found in {corpus_text_dir}")
        return []

    file_list = "\n".join(f"  - {f.name}" for f in txt_files)
    seed_file = txt_files[0].name
    chains: List[dict] = []

    remaining = n_chains
    while remaining > 0:
        batch = min(batch_size, remaining)
        prompt = _render(
            gen_template,
            FILE_LIST=file_list,
            CATEGORY_NAME=category["name"],
            CATEGORY_DESCRIPTION=category["description"].strip(),
            N_PAIRS=str(batch),
        )

        reply = await _run_opencode(prompt, corpus_text_dir, opencode_bin)
        pairs = _extract_json_array(reply)

        if pairs:
            new_chains = _pairs_to_chains(pairs, category, seed_file)
            chains.extend(new_chains)
            remaining -= len(new_chains)
            print(f"    [{category['name']}] +{len(new_chains)} pairs (total {len(chains)}/{n_chains})")
        else:
            print(f"    [{category['name']}] WARNING: could not parse JSON from reply (len={len(reply)})")
            if reply:
                print(f"    [{category['name']}] Reply preview: {reply[:300]}")
            # Skip this batch to avoid infinite loop
            remaining -= batch

    return chains[:n_chains]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Q&A chains via opencode run")
    parser.add_argument("--corpus_text_dir", default="corpus_text",
                        help="Directory of .txt files for OpenCode to grep/read")
    parser.add_argument("--output", default="qa_chains_raw.json")
    parser.add_argument("--n_chains", type=int, default=200,
                        help="Total chains to generate across all categories")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Q&A pairs to request per opencode run invocation")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Max concurrent opencode run processes")
    parser.add_argument("--opencode", default=None,
                        help="Path to opencode binary (auto-detected if omitted)")
    parser.add_argument("--categories_cfg", default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    opencode_bin = args.opencode or _find_opencode()
    corpus_text_dir = str(Path(args.corpus_text_dir).resolve())

    # Ensure opencode.json is in the corpus dir so --dir picks it up
    src_cfg = PROJECT_DIR / "opencode.json"
    dst_cfg = Path(corpus_text_dir) / "opencode.json"
    if src_cfg.exists() and (not dst_cfg.exists() or src_cfg.stat().st_mtime > dst_cfg.stat().st_mtime):
        shutil.copy2(src_cfg, dst_cfg)

    # Load categories
    cfg_path = args.categories_cfg or str(CATEGORIES_CFG)
    with open(cfg_path) as f:
        cat_cfg = yaml.safe_load(f)
    categories: List[dict] = cat_cfg["categories"]

    # Sharding
    rank_categories = [c for i, c in enumerate(categories) if i % args.world_size == args.rank]
    if not rank_categories:
        print(f"Rank {args.rank}: no categories assigned.")
        return

    if args.world_size > 1:
        args.output = args.output.replace(".json", f"_part{args.rank}.json")

    # Resume
    existing: List[dict] = []
    done_categories: set = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            try:
                existing = json.load(f)
                done_categories = {c["category"] for c in existing}
                print(f"Resuming: {len(existing)} chains already saved, "
                      f"done categories: {done_categories}")
            except json.JSONDecodeError:
                existing = []

    rank_categories = [c for c in rank_categories if c["name"] not in done_categories]
    if not rank_categories:
        print("All categories complete.")
        return

    per_category = max(1, args.n_chains // len(categories))
    gen_template = _load_prompt("qa_gen_agent.txt")

    print(f"Generating {per_category} chains × {len(rank_categories)} categories "
          f"(concurrency={args.concurrency}, opencode={opencode_bin})")

    semaphore = asyncio.Semaphore(args.concurrency)
    results = list(existing)

    async def _process(cat: dict) -> List[dict]:
        async with semaphore:
            print(f"\n→ Category: {cat['name']} ({per_category} pairs)")
            chains = await generate_for_category(
                category=cat,
                corpus_text_dir=corpus_text_dir,
                n_chains=per_category,
                opencode_bin=opencode_bin,
                batch_size=args.batch_size,
                gen_template=gen_template,
            )
            return chains

    tasks = [_process(cat) for cat in rank_categories]
    for coro in asyncio.as_completed(tasks):
        cat_chains = await coro
        results.extend(cat_chains)
        _atomic_save(results, args.output)

    print(f"\nDone. {len(results)} total chains → {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

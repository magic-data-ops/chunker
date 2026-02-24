#!/usr/bin/env python3
"""Generate Q&A evaluation chains using Claude Code CLI as the agent harness.

Claude Code's native Grep, Read, and Glob tools navigate the text corpus
autonomously.  For each category we run `claude -p` from the corpus text
directory, parse the stream-json event stream, and extract the structured
Q&A pairs along with tool-call provenance.

No server needed — each invocation is a standalone `claude` subprocess.

Usage:
    python generate_qa_chains.py --corpus_text_dir ./corpus_text --output qa_chains_raw.json
    python generate_qa_chains.py --corpus_text_dir ./corpus_text --n_chains 200 --batch_size 5
    python generate_qa_chains.py --rank 0 --world_size 4  # sharding (categories level)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

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

CORPUS_CLAUDE_MD_TEMPLATE = """\
# Corpus Search Instructions

This directory contains text documents from a document corpus.
You are generating Q&A pairs for LLM benchmarking.

## Available Tools
- Grep: search for phrases across all .txt files
- Read: read file contents with line numbers
- Glob: list files matching patterns

## Search Strategy
- Use Grep with 2-4 word phrases (not single words)
- After finding matches, use Read to get full context
- For multi-hop questions, chain searches across different files
- Never guess or fabricate content — only use what you find in these files

## File Structure
- All files are .txt format named doc_XXXX_<title>.txt
- Total corpus: {n_files} files
"""


# ---------------------------------------------------------------------------
# Run envelope
# ---------------------------------------------------------------------------


@dataclass
class ClaudeRunResult:
    """Structured result from a single `claude -p` invocation."""
    reply_text: str = ""                  # Final text response (Q/A JSON string)
    reply_json: Optional[list] = None     # Parsed JSON array if possible, else None
    tool_events: list = field(default_factory=list)   # [{tool, input, result, tool_use_id}]
    reasoning_blocks: list = field(default_factory=list)  # Intermediate assistant text blocks
    subagent_events: list = field(default_factory=list)  # Task tool invocations (subagents)
    raw_stdout: str = ""                  # Raw CLI output for debugging
    meta: dict = field(default_factory=dict)  # {model, max_budget, duration_ms, exit_code, cost_usd, ...}
    errors: list = field(default_factory=list)  # Any errors encountered during parsing


# ---------------------------------------------------------------------------
# Provenance dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ToolProvenance:
    """Per-tool-call provenance record."""
    tool: str
    tool_use_id: str
    file_path: Optional[str] = None
    line_range: Optional[Tuple[int, int]] = None
    content_length: int = 0
    content_hash: str = ""
    snippet_preview: str = ""
    # Grep-specific
    grep_pattern: Optional[str] = None
    grep_hits: Optional[list] = None  # [{file, line_no, text}]


@dataclass
class ProvenanceReport:
    """Aggregate provenance from all tool calls in a run."""
    tool_provenances: list = field(default_factory=list)   # [ToolProvenance, ...]
    files_accessed: list = field(default_factory=list)      # [{file, line_range, content_length}]
    grep_queries: list = field(default_factory=list)        # [{pattern, files_hit}]
    unique_files: list = field(default_factory=list)
    total_content_read_chars: int = 0


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
# Claude Code CLI wrapper
# ---------------------------------------------------------------------------


def _coerce_str(value) -> str:
    """Coerce a value to string. Handles list/dict content blocks from Claude."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _parse_stream_json(stdout: str) -> Tuple[str, Optional[list], list, dict, list]:
    """Parse Claude Code stream-json output.

    Returns (reply_text, reply_json, tool_events, meta, reasoning_blocks).
    """
    text_parts: list[str] = []
    tool_events: list[dict] = []
    pending_tools: dict[str, dict] = {}  # tool_use_id -> partial event
    meta: dict = {}
    result_text_from_event: Optional[str] = None

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type")

        if etype == "system" and event.get("subtype") == "init":
            meta["session_id"] = event.get("session_id")
            meta["cli_version"] = event.get("claude_code_version")
            meta["model_from_init"] = event.get("model")

        elif etype == "assistant":
            msg = event.get("message", {})
            for block in msg.get("content", []):
                btype = block.get("type")
                if btype == "text" and "text" in block:
                    text_parts.append(block["text"])
                elif btype == "tool_use":
                    tool_use_id = block.get("id")
                    tool_name = block.get("name")
                    if not tool_use_id or not tool_name:
                        continue
                    pending_tools[tool_use_id] = {
                        "tool": tool_name,
                        "input": block.get("input", {}),
                        "tool_use_id": tool_use_id,
                        "result": None,
                    }

        elif etype == "user":
            msg = event.get("message", {})
            for block in msg.get("content", []):
                if block.get("type") == "tool_result":
                    tid = block.get("tool_use_id")
                    result_content = _coerce_str(block.get("content", ""))
                    if tid and tid in pending_tools:
                        pending_tools[tid]["result"] = result_content
                    # Also capture structured metadata from tool_use_result
                    tur = event.get("tool_use_result", {})
                    if tid and tid in pending_tools and tur:
                        pending_tools[tid]["result_meta"] = tur

        elif etype == "result":
            meta["duration_ms"] = event.get("duration_ms")
            meta["cost_usd"] = event.get("total_cost_usd")
            meta["num_turns"] = event.get("num_turns")
            meta["is_error"] = event.get("is_error", False)
            result_text_from_event = event.get("result", "")

    tool_events = list(pending_tools.values())

    # Prefer the canonical result event as source of truth for the reply.
    # Intermediate assistant text blocks may contain partial/mixed content
    # that is less reliable for JSON extraction.
    if result_text_from_event:
        reply_text = result_text_from_event
    else:
        reply_text = "\n".join(text_parts)

    # Try to parse reply as JSON array
    reply_json = _extract_json_array(reply_text)

    return reply_text, reply_json, tool_events, meta, text_parts


async def _run_claude_code(
    prompt: str,
    corpus_dir: str,
    claude_bin: str = "claude",
    model: str = "sonnet",
    max_budget_usd: float = 1.00,
    timeout: float = 900.0,
    system_prompt: Optional[str] = None,
    save_raw_path: Optional[str] = None,
) -> ClaudeRunResult:
    """Run `claude -p` and return a structured ClaudeRunResult."""
    cmd = [
        claude_bin, "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--model", model,
        "--max-budget-usd", str(max_budget_usd),
        "--allowedTools", "Read", "Grep", "Glob", "Bash(ls:*)", "Bash(wc:*)", "Task",
    ]
    if system_prompt:
        cmd += ["--system-prompt", system_prompt]

    # Filter out CLAUDECODE env var to avoid nested-session error
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=corpus_dir,
        env=env,
    )

    result = ClaudeRunResult()
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        result.errors.append("timeout")
        result.meta["exit_code"] = -1
        return result

    raw_stdout = stdout_bytes.decode(errors="replace")
    result.raw_stdout = raw_stdout
    result.meta["exit_code"] = proc.returncode

    # Save raw output if requested
    if save_raw_path:
        parent = os.path.dirname(save_raw_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(save_raw_path, "w", encoding="utf-8") as f:
            f.write(raw_stdout)

    if proc.returncode != 0:
        stderr_text = stderr_bytes.decode(errors="replace")
        result.errors.append(f"exit_code_{proc.returncode}")
        if stderr_text:
            result.errors.append(f"stderr: {stderr_text[:500]}")

    # Parse the stream-json output
    reply_text, reply_json, tool_events, meta, reasoning_blocks = _parse_stream_json(raw_stdout)
    result.reply_text = reply_text
    result.reply_json = reply_json
    result.tool_events = tool_events
    result.reasoning_blocks = reasoning_blocks
    result.subagent_events = [e for e in tool_events if e.get("tool") == "Task"]
    result.meta.update(meta)
    result.meta["model"] = model
    result.meta["max_budget_usd"] = max_budget_usd
    result.meta["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Truncation detection
    if reply_text and not reply_json:
        # Check for unmatched brackets
        open_brackets = reply_text.count("[") - reply_text.count("]")
        open_braces = reply_text.count("{") - reply_text.count("}")
        if open_brackets > 0 or open_braces > 0:
            result.errors.append("truncated_output")

    # Check for incomplete tool calls (tool_use without matching tool_result)
    incomplete_tools = [e for e in tool_events if e.get("result") is None]
    if incomplete_tools:
        result.errors.append(f"incomplete_tool_calls:{len(incomplete_tools)}")

    return result


# ---------------------------------------------------------------------------
# Provenance extraction
# ---------------------------------------------------------------------------


def _parse_read_line_range(result_text: str) -> Optional[Tuple[int, int]]:
    """Extract (min_line, max_line) from Read tool output with line number prefixes."""
    lines = []
    for line in result_text.splitlines():
        # Match both "  1520\t" and "     1->" formats
        match = re.match(r'\s*(\d+)[→\t]', line)
        if match:
            lines.append(int(match.group(1)))
    if lines:
        return (min(lines), max(lines))
    return None


def _parse_grep_hits(result_text: str) -> list:
    """Parse Grep tool result into [{file, line_no, text}]."""
    hits = []
    for line in result_text.splitlines():
        # Match "file.txt:123: some text" or "file.txt:123:some text"
        match = re.match(r'([^:]+):(\d+):\s*(.*)', line)
        if match:
            hits.append({
                "file": match.group(1).strip(),
                "line_no": int(match.group(2)),
                "text": match.group(3).strip(),
            })
    return hits


def _extract_provenance(tool_events: list) -> ProvenanceReport:
    """Build a ProvenanceReport from Claude Code tool events."""
    report = ProvenanceReport()
    seen_files = set()
    total_chars = 0

    for event in tool_events:
        tool = event.get("tool", "")
        tool_use_id = event.get("tool_use_id", "")
        inp = event.get("input", {})
        result = _coerce_str(event.get("result", ""))
        result_meta = event.get("result_meta", {})

        content_len = len(result)
        total_chars += content_len
        content_hash = hashlib.sha1(result.encode(errors="replace")).hexdigest()[:12]
        preview = result[:300]

        prov = ToolProvenance(
            tool=tool,
            tool_use_id=tool_use_id,
            content_length=content_len,
            content_hash=content_hash,
            snippet_preview=preview,
        )

        if tool == "Read":
            fp = inp.get("file_path", "")
            prov.file_path = fp
            if fp:
                seen_files.add(os.path.basename(fp))
            prov.line_range = _parse_read_line_range(result)
            # Also try structured metadata
            file_meta = result_meta.get("file", {}) if isinstance(result_meta, dict) else {}
            if file_meta and not prov.line_range:
                start = file_meta.get("startLine")
                num = file_meta.get("numLines")
                if start is not None and num is not None:
                    prov.line_range = (start, start + num - 1)
            if prov.file_path and prov.line_range:
                report.files_accessed.append({
                    "file": os.path.basename(prov.file_path),
                    "line_range": prov.line_range,
                    "content_length": content_len,
                })

        elif tool == "Grep":
            pattern = inp.get("pattern", "")
            prov.grep_pattern = pattern
            hits = _parse_grep_hits(result)
            prov.grep_hits = hits
            hit_files = list({h["file"] for h in hits})
            for hf in hit_files:
                seen_files.add(os.path.basename(hf))
            report.grep_queries.append({
                "pattern": pattern,
                "files_hit": hit_files,
            })

        elif tool == "Glob":
            # Extract file list from result
            for line in result.splitlines():
                line = line.strip()
                if line:
                    seen_files.add(os.path.basename(line))

        report.tool_provenances.append(prov)

    report.unique_files = sorted(seen_files)
    report.total_content_read_chars = total_chars
    return report


# ---------------------------------------------------------------------------
# Context span (provenance-based)
# ---------------------------------------------------------------------------


def _compute_run_span(provenance: ProvenanceReport) -> int:
    """Compute the max within-file line span across all Read tool calls.

    Returns the max within-file line span across all Read tool calls.
    Returns 0 when no Read tool calls with line ranges were recorded.
    """
    per_file_ranges: dict[str, list[int]] = {}
    for acc in provenance.files_accessed:
        fname = acc["file"]
        lr = acc.get("line_range")
        if lr:
            per_file_ranges.setdefault(fname, []).extend([lr[0], lr[1]])

    if not per_file_ranges:
        return 0

    max_span = 0
    for fname, line_nums in per_file_ranges.items():
        span = max(line_nums) - min(line_nums)
        if span > max_span:
            max_span = span

    return max_span


def _compute_pair_span(pair: dict, provenance: Optional[ProvenanceReport] = None) -> int:
    """Compute context span in lines for a single Q/A pair.

    Priority:
    1. Pair's evidence_locations (agent-reported file+line per evidence snippet)
    2. Run-level provenance (fallback — less precise for batches)

    Returns 0 when no line-range data is available.
    """
    locs = pair.get("evidence_locations")
    if locs and len(locs) >= 1:
        all_lines = []
        for loc in locs:
            s = loc.get("start_line")
            e = loc.get("end_line")
            if s is not None and e is not None:
                all_lines.extend([s, e])
        if all_lines:
            return max(all_lines) - min(all_lines)
    # Fallback: run-level provenance
    if provenance:
        return _compute_run_span(provenance)
    return 0


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
# Pair validation
# ---------------------------------------------------------------------------


def _validate_pair(pair: dict, category: dict,
                   min_answer_len: int = 20,
                   min_evidence_len: int = 20) -> Tuple[bool, str]:
    """Validate a single Q&A pair against quality and category constraints.

    Returns (is_valid, reason).
    """
    q = pair.get("question", "").strip()
    a = pair.get("golden_answer", "").strip()
    evidence = pair.get("evidence_snippets", [])

    # Auto-fix: append question mark if question looks valid but is missing one
    if q and not q.endswith("?") and len(q) > 30:
        q = q.rstrip(".") + "?"
        pair["question"] = q

    # Fall back to entity-level evidence if top-level is empty
    if not evidence:
        entities = pair.get("entities", [])
        for ent in entities:
            snip = ent.get("evidence_snippet", "")
            if isinstance(snip, str) and snip.strip():
                evidence.append(snip)
        if evidence:
            pair["evidence_snippets"] = evidence

    if not q.endswith("?"):
        return False, "question does not end with '?'"

    if len(a) < min_answer_len:
        return False, f"golden_answer too short ({len(a)} < {min_answer_len} chars)"

    if not evidence:
        return False, "no evidence_snippets"

    for i, snip in enumerate(evidence):
        if not isinstance(snip, str) or len(snip.strip()) < min_evidence_len:
            return False, f"evidence_snippet[{i}] too short ({len(snip.strip()) if isinstance(snip, str) else 0} < {min_evidence_len} chars)"

    # Hop constraint enforcement
    min_hops = category.get("min_hops", 1)
    max_hops = category.get("max_hops", 4)
    n_evidence = len(evidence)
    if n_evidence < min_hops:
        return False, f"too few evidence snippets ({n_evidence} < min_hops={min_hops})"
    if n_evidence > max_hops:
        return False, f"too many evidence snippets ({n_evidence} > max_hops={max_hops})"

    # Multi-hop requires at least 2 evidence snippets
    if category.get("name") == "multi_hop_reasoning" and n_evidence < 2:
        return False, "multi_hop_reasoning requires >= 2 evidence snippets"

    return True, "ok"


# ---------------------------------------------------------------------------
# Seed context from corpus files (no chunk store needed)
# ---------------------------------------------------------------------------


def _sample_seed_context(corpus_text_dir: str, rng: random.Random,
                         max_chars: int = 500) -> tuple[str, str]:
    """Return (context_snippet, filename) from a random corpus file."""
    txt_files = sorted(Path(corpus_text_dir).glob("*.txt"))
    if not txt_files:
        return "", ""
    chosen_file = rng.choice(txt_files)
    try:
        text = chosen_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", chosen_file.name
    if len(text) <= max_chars:
        return text, chosen_file.name
    start = rng.randint(0, len(text) - max_chars)
    # Snap to nearest newline to avoid cutting mid-sentence
    nl = text.find("\n", start)
    if nl != -1 and nl < start + 100:
        start = nl + 1
    return text[start:start + max_chars], chosen_file.name


# ---------------------------------------------------------------------------
# Chain conversion
# ---------------------------------------------------------------------------


def _pairs_to_chains(pairs: List[dict], category: dict, prompt_seed_file: str,
                     provenance: Optional[ProvenanceReport] = None) -> List[dict]:
    """Convert the agent's raw JSON pairs into the canonical chain schema."""
    chains = []
    for pair in pairs:
        q = pair.get("question", "").strip()
        a = pair.get("golden_answer", "").strip()
        if not q or not a:
            continue
        evidence = pair.get("evidence_snippets", [])
        sources = pair.get("source_files") or [prompt_seed_file]

        padded_sources = sources + [sources[-1]] * max(0, len(evidence) - len(sources))

        # Build per-hop partial answers from entities or golden_answer
        entities = pair.get("entities", [])
        entity_descriptions = [e.get("description", "") for e in entities if e.get("description")]

        hop_path = []
        for i, (snip, src) in enumerate(zip(evidence, padded_sources)):
            # Use entity description if available, otherwise golden_answer for last hop
            if i < len(entity_descriptions):
                partial = entity_descriptions[i]
            elif i == len(evidence) - 1:
                partial = a  # golden_answer
            else:
                partial = snip
            hop_path.append({
                "hop_index": i,
                "chunk_id": f"{src}:evidence_{i}",
                "chunk_text": snip,
                "partial_answer": partial,
                "retrieval_score": None,
            })

        chain = {
            "chain_id": str(uuid.uuid4()),
            "category": category["name"],
            "source_file": sources[0] if sources else "unknown",
            "prompt_seed_file": prompt_seed_file,
            "question": q,
            "final_answer": a,
            "hop_path": hop_path,
            "hop_count": max(1, len(hop_path)),
            "termination_reason": "agent_complete",
            "single_answer_heuristic": category.get("max_hops", 2) == 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Attach evidence_locations from the agent's output if present
        evidence_locations = pair.get("evidence_locations")
        if evidence_locations:
            chain["evidence_locations"] = evidence_locations

        # Preserve rich schema fields (entity disambiguation, etc.)
        for key in ("difficulty", "entities", "disambiguation_statement"):
            if key in pair:
                chain[key] = pair[key]

        # Attach provenance report if available
        if provenance:
            chain["provenance_report"] = {
                "unique_files": provenance.unique_files,
                "files_accessed": provenance.files_accessed,
                "grep_queries": provenance.grep_queries,
                "tool_call_count": len(provenance.tool_provenances),
                "total_content_read_chars": provenance.total_content_read_chars,
            }
            chain["context_span_lines"] = _compute_pair_span(pair, provenance)

        chains.append(chain)
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
# Run logging
# ---------------------------------------------------------------------------


def _save_run_log(
    result: ClaudeRunResult,
    provenance: Optional[ProvenanceReport],
    category_name: str,
    run_id: str,
    prompt_seed_file: str,
    pairs_generated: int,
    pairs_valid: int,
    log_dir: str,
) -> None:
    """Save a structured JSON log for a single claude -p run."""
    os.makedirs(log_dir, exist_ok=True)

    # Build tool call summaries
    tool_calls = []
    for event in result.tool_events:
        result_text = _coerce_str(event.get("result", ""))
        entry = {
            "tool": event.get("tool", ""),
            "input": event.get("input", {}),
            "result_preview": result_text[:500],
            "result_length": len(result_text),
        }
        # Add parsed line range for Read calls
        if event.get("tool") == "Read" and result_text:
            lr = _parse_read_line_range(result_text)
            if lr:
                entry["line_range"] = list(lr)
        tool_calls.append(entry)

    # Build provenance summary
    prov_summary = None
    if provenance:
        prov_summary = {
            "unique_files": provenance.unique_files,
            "grep_queries": provenance.grep_queries,
            "total_content_read_chars": provenance.total_content_read_chars,
        }

    # Extract subagent calls
    subagent_calls = []
    for event in result.subagent_events:
        inp = event.get("input", {})
        subagent_calls.append({
            "subagent_type": inp.get("subagent_type", ""),
            "description": inp.get("description", ""),
            "result_preview": _coerce_str(event.get("result", ""))[:500],
        })

    log_entry = {
        "run_id": run_id,
        "category": category_name,
        "timestamp": result.meta.get("timestamp", ""),
        "model": result.meta.get("model", ""),
        "cost_usd": result.meta.get("cost_usd"),
        "duration_ms": result.meta.get("duration_ms"),
        "prompt_seed_file": prompt_seed_file,
        "reasoning": result.reasoning_blocks,
        "tool_calls": tool_calls,
        "subagent_calls": subagent_calls,
        "provenance_summary": prov_summary,
        "pairs_generated": pairs_generated,
        "pairs_valid": pairs_valid,
        "errors": result.errors,
    }

    log_path = os.path.join(log_dir, f"{category_name}_{run_id}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Corpus CLAUDE.md generation
# ---------------------------------------------------------------------------


def _ensure_corpus_claude_md(corpus_text_dir: str) -> None:
    """Generate a CLAUDE.md in the corpus directory for Claude Code context.

    Skips if the file already exists (user-maintained).
    """
    md_path = Path(corpus_text_dir) / "CLAUDE.md"
    if md_path.exists():
        return
    txt_files = sorted(Path(corpus_text_dir).glob("*.txt"))
    content = CORPUS_CLAUDE_MD_TEMPLATE.format(n_files=len(txt_files))
    md_path.write_text(content, encoding="utf-8")


def _corpus_stem(corpus_text_dir: str) -> str:
    """Derive corpus name from .txt files in the directory.

    Single .txt file → use its stem (e.g. "enron_complete").
    Multiple files → use directory name.
    """
    txt_files = sorted(Path(corpus_text_dir).glob("*.txt"))
    if len(txt_files) == 1:
        return txt_files[0].stem
    return Path(corpus_text_dir).name or "corpus"


# ---------------------------------------------------------------------------
# Per-category generation
# ---------------------------------------------------------------------------


async def generate_for_category(
    category: dict,
    corpus_text_dir: str,
    n_chains: int,
    claude_bin: str,
    model: str,
    max_budget_usd: float,
    batch_size: int,
    gen_template: str,
    max_retries: int = 2,
    min_answer_len: int = 20,
    min_evidence_len: int = 20,
    rng: Optional[random.Random] = None,
    save_raw_runs: bool = False,
    output_dir: str = ".",
    log_dir: Optional[str] = None,
) -> List[dict]:
    """Generate n_chains Q&A pairs for one category via `claude -p`."""
    txt_files = sorted(Path(corpus_text_dir).glob("*.txt"))
    if not txt_files:
        print(f"  [WARN] No .txt files found in {corpus_text_dir}")
        return []

    file_list = "\n".join(f"  - {f.name}" for f in txt_files)
    chains: List[dict] = []

    if rng is None:
        rng = random.Random()

    # Safety valve: cap total attempts
    max_attempts = max_retries * math.ceil(n_chains / batch_size) + 3
    attempt_count = 0
    consecutive_zero_progress = 0

    remaining = n_chains
    while remaining > 0:
        if attempt_count >= max_attempts:
            print(f"    [{category['name']}] Safety valve: reached {max_attempts} attempts, stopping.")
            break
        if consecutive_zero_progress >= 3:
            print(f"    [{category['name']}] No-progress guard: 3 consecutive batches with 0 valid pairs, stopping.")
            break

        attempt_count += 1
        batch = min(batch_size, remaining)

        # Sample seed context directly from corpus files
        seed_context, prompt_seed_file = _sample_seed_context(corpus_text_dir, rng)
        seed_entity = ""

        prompt = _render(
            gen_template,
            FILE_LIST=file_list,
            CATEGORY_NAME=category["name"],
            CATEGORY_DESCRIPTION=category["description"].strip(),
            N_PAIRS=str(batch),
            SEED_CONTEXT=seed_context,
            SEED_ENTITY=seed_entity,
        )

        # Determine raw output path
        run_id = str(uuid.uuid4())[:8]
        save_raw_path = None
        if save_raw_runs:
            raw_dir = os.path.join(output_dir, "raw_runs")
            save_raw_path = os.path.join(raw_dir, f"{category['name']}_{run_id}.jsonl")

        result = await _run_claude_code(
            prompt, corpus_dir=corpus_text_dir,
            claude_bin=claude_bin, model=model,
            max_budget_usd=max_budget_usd,
            save_raw_path=save_raw_path,
        )

        # Check for run errors
        if result.errors:
            print(f"    [{category['name']}] Run errors: {result.errors}")
            if "timeout" in result.errors:
                remaining -= batch
                consecutive_zero_progress += 1
                continue

        pairs = result.reply_json or _extract_json_array(result.reply_text)

        if pairs is None:
            # Parse failure -- retry
            print(f"    [{category['name']}] WARNING: could not parse JSON from reply (len={len(result.reply_text)})")
            if result.reply_text:
                print(f"    [{category['name']}] Reply preview: {result.reply_text[:300]}")
            # Retry up to max_retries for parse failures
            if attempt_count <= max_retries:
                print(f"    [{category['name']}] Retrying... ({attempt_count}/{max_retries})")
                continue
            # All retries exhausted for parse failure
            remaining -= batch
            consecutive_zero_progress += 1
            continue

        # Extract provenance from tool calls
        provenance = _extract_provenance(result.tool_events)

        # Validate pairs
        valid_pairs = []
        for pair in pairs:
            ok, reason = _validate_pair(pair, category,
                                        min_answer_len=min_answer_len,
                                        min_evidence_len=min_evidence_len)
            if ok:
                valid_pairs.append(pair)
            else:
                print(f"    [{category['name']}] Rejected pair: {reason}")

        if valid_pairs:
            new_chains = _pairs_to_chains(valid_pairs, category, prompt_seed_file,
                                          provenance=provenance)
            chains.extend(new_chains)
            remaining -= len(new_chains)
            consecutive_zero_progress = 0
            print(f"    [{category['name']}] +{len(new_chains)} valid pairs "
                  f"(total {len(chains)}/{n_chains})")
        else:
            # Agent understood format but content was bad -- don't retry
            print(f"    [{category['name']}] 0 valid pairs from {len(pairs)} parsed. "
                  f"Skipping batch.")
            remaining -= batch
            consecutive_zero_progress += 1

        # Save structured run log
        if log_dir:
            _save_run_log(
                result=result,
                provenance=provenance,
                category_name=category["name"],
                run_id=run_id,
                prompt_seed_file=prompt_seed_file,
                pairs_generated=len(pairs) if pairs else 0,
                pairs_valid=len(valid_pairs),
                log_dir=log_dir,
            )

    return chains[:n_chains]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Deliverable export
# ---------------------------------------------------------------------------


def _chain_to_deliverable_sample(chain: dict) -> dict:
    """Convert an internal chain to the deliverable sample schema."""
    # relevant_context: join evidence snippets
    evidence = chain.get("hop_path", [])
    context_parts = [hop.get("chunk_text", "") for hop in evidence if hop.get("chunk_text")]
    if not context_parts:
        # Fallback to evidence_snippets on the chain directly
        for key in ("evidence_snippets",):
            if key in chain and chain[key]:
                context_parts = chain[key]
                break

    separator = "\n\n" + "." * 50 + "\n\n"
    relevant_context = separator.join(context_parts)

    # context_location_in_file: from evidence_locations or hop path
    locations = []
    if chain.get("evidence_locations"):
        for loc in chain["evidence_locations"]:
            if loc.get("file") and loc.get("start_line") is not None and loc.get("end_line") is not None:
                locations.append({
                    "file": loc["file"],
                    "start_line": loc["start_line"],
                    "end_line": loc["end_line"],
                })
    # Fallback: entity-level locations
    if not locations:
        for ent in chain.get("entities", []):
            eloc = ent.get("evidence_location", {})
            if eloc.get("file") and eloc.get("start_line") is not None and eloc.get("end_line") is not None:
                locations.append({
                    "file": eloc["file"],
                    "start_line": eloc["start_line"],
                    "end_line": eloc["end_line"],
                })

    sample = {
        "relevant_context": relevant_context,
        "context_location_in_file": locations,
        "suggested_prompt": chain.get("question", ""),
        "golden_response": chain.get("final_answer", ""),
        "num_turns": chain.get("num_turns", 1),
        "conversation_history": chain.get("conversation_history", []),
    }
    return sample


def _validate_deliverable_sample(sample: dict) -> Tuple[bool, str]:
    """Validate a single deliverable sample. Returns (ok, reason)."""
    if not sample.get("relevant_context", "").strip():
        return False, "empty relevant_context"
    if not sample.get("context_location_in_file"):
        return False, "no context_location_in_file"
    if not sample.get("suggested_prompt", "").strip():
        return False, "empty suggested_prompt"
    if not sample.get("golden_response", "").strip():
        return False, "empty golden_response"
    # Validate multi-turn fields if present
    num_turns = sample.get("num_turns", 1)
    history = sample.get("conversation_history", [])
    if num_turns > 1:
        if len(history) != num_turns - 1:
            return False, f"conversation_history length {len(history)} != num_turns-1 ({num_turns - 1})"
        for i, turn in enumerate(history):
            if not isinstance(turn, dict):
                return False, f"conversation_history turn {i} is not a dict"
            if not turn.get("user", "").strip() or not turn.get("assistant", "").strip():
                return False, f"conversation_history turn {i} has empty user or assistant"
    return True, "ok"


def _build_grouped_deliverable(
    chains: List[dict],
    categories: List[dict],
    samples_per_category: int,
) -> Tuple[dict, List[str]]:
    """Build the grouped deliverable from internal chains.

    Returns (deliverable_dict, errors_list).
    Errors are category-level failures (e.g. wrong count).
    """
    # Group chains by category
    by_cat: dict[str, list] = {}
    for chain in chains:
        cat_name = chain.get("category", "")
        by_cat.setdefault(cat_name, []).append(chain)

    # Build category lookup for display_name
    cat_lookup = {c["name"]: c for c in categories}

    errors: List[str] = []
    category_blocks: List[dict] = []

    for cat in categories:
        cat_name = cat["name"]
        cat_chains = by_cat.get(cat_name, [])

        if len(cat_chains) < samples_per_category:
            errors.append(
                f"Category '{cat_name}': only {len(cat_chains)} samples, "
                f"need {samples_per_category}"
            )
            continue

        # Take exactly samples_per_category
        selected = cat_chains[:samples_per_category]
        samples = []
        for chain in selected:
            sample = _chain_to_deliverable_sample(chain)
            ok, reason = _validate_deliverable_sample(sample)
            if not ok:
                errors.append(f"Category '{cat_name}': invalid sample — {reason}")
                continue
            samples.append(sample)

        if len(samples) != samples_per_category:
            errors.append(
                f"Category '{cat_name}': only {len(samples)} valid samples "
                f"after validation, need {samples_per_category}"
            )
            continue

        category_blocks.append({
            "category_id": cat_name,
            "category_display_name": cat.get("display_name", cat_name),
            "samples": samples,
        })

    domain_scopes = {c.get("domain_scope", "unknown") for c in categories}
    domain_scope = domain_scopes.pop() if len(domain_scopes) == 1 else "mixed"

    deliverable = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain_scope": domain_scope,
        "categories": category_blocks,
    }
    return deliverable, errors


def _atomic_save_dict(data: dict, path: str) -> None:
    """Atomic save for a dict (deliverable format)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)


def _export_deliverable_csv(deliverable: dict, csv_path: str) -> None:
    """Export the grouped deliverable to CSV with one row per sample.

    Columns: description, relevant_context, context_location_in_file,
             template_question, golden_response, num_turns, conversation_history
    """
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = csv_path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "description",
            "relevant_context",
            "context_location_in_file",
            "template_question",
            "golden_response",
            "num_turns",
            "conversation_history",
        ])
        for cat in deliverable.get("categories", []):
            display_name = cat.get("category_display_name", cat.get("category_id", ""))
            for sample in cat.get("samples", []):
                writer.writerow([
                    display_name,
                    sample.get("relevant_context", ""),
                    json.dumps(sample.get("context_location_in_file", []), indent=12),
                    sample.get("suggested_prompt", ""),
                    sample.get("golden_response", ""),
                    sample.get("num_turns", 1),
                    json.dumps(sample.get("conversation_history", []), indent=2),
                ])
    if os.path.exists(csv_path):
        os.remove(csv_path)
    os.rename(tmp, csv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Q&A chains via Claude Code CLI")
    parser.add_argument("--corpus_text_dir", default="corpus_text",
                        help="Directory of .txt files for Claude Code to Grep/Read")
    parser.add_argument("--output", default="qa_chains_raw.json")
    parser.add_argument("--n_chains", type=int, default=None,
                        help="Total chains to generate (legacy; overridden by --samples-per-category)")
    parser.add_argument("--samples-per-category", type=int, default=3,
                        help="Exact number of samples to generate per category (default: 3)")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Q&A pairs to request per Claude Code invocation")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Max concurrent Claude Code processes")
    parser.add_argument("--claude-bin", default="claude",
                        help="Path to claude binary (default: 'claude')")
    parser.add_argument("--model", default="sonnet",
                        choices=["sonnet", "opus", "haiku"],
                        help="Claude model to use (default: sonnet)")
    parser.add_argument("--max-budget-usd", type=float, default=1.00,
                        help="Cost cap per Claude Code invocation in USD")
    parser.add_argument("--categories_cfg", default=None)
    parser.add_argument("--prompt-template", default=None,
                        help="Path to agent prompt template (default: prompts/qa_gen_agent.txt)")
    parser.add_argument("--max_retries", type=int, default=2,
                        help="Max retries per batch on parse failure")
    parser.add_argument("--min_answer_len", type=int, default=20,
                        help="Minimum golden_answer length in chars")
    parser.add_argument("--min_evidence_len", type=int, default=20,
                        help="Minimum evidence_snippet length in chars")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--save-raw-runs", action="store_true",
                        help="Save per-run raw JSONL output for debugging")
    parser.add_argument("--log-dir", default="logs",
                        help="Directory for structured run logs (default: logs)")
    parser.add_argument("--deliverable-output", default="qa_deliverable_grouped.json",
                        help="Path for grouped deliverable export")
    parser.add_argument("--csv-output", default="qa_deliverable.csv",
                        help="Path for CSV deliverable export")
    parser.add_argument("--export-deliverable", action="store_true", default=True,
                        help="Export grouped deliverable after generation (default: true)")
    parser.add_argument("--no-export-deliverable", action="store_false", dest="export_deliverable",
                        help="Disable grouped deliverable export")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    claude_bin = args.claude_bin
    model = args.model
    max_budget_usd = args.max_budget_usd
    corpus_text_dir = str(Path(args.corpus_text_dir).resolve())

    # Dynamic output naming: {corpus_stem}_{YYYYMMDD}_*
    stem = _corpus_stem(corpus_text_dir)
    ts = datetime.now().strftime("%Y%m%d")
    if args.output == "qa_chains_raw.json":
        args.output = f"{stem}_{ts}_qa_chains_raw.json"
    if args.deliverable_output == "qa_deliverable_grouped.json":
        args.deliverable_output = f"{stem}_{ts}_qa_deliverable_grouped.json"
    if args.csv_output == "qa_deliverable.csv":
        args.csv_output = f"{stem}_{ts}_qa_deliverable.csv"

    output_dir = str(Path(args.output).parent.resolve()) if os.path.dirname(args.output) else "."

    rng = random.Random(args.seed) if args.seed is not None else random.Random()

    # Generate CLAUDE.md in corpus dir for Claude Code context
    _ensure_corpus_claude_md(corpus_text_dir)

    # Load categories
    cfg_path = args.categories_cfg or str(CATEGORIES_CFG)
    with open(cfg_path) as f:
        cat_cfg = yaml.safe_load(f)
    categories: List[dict] = cat_cfg["categories"]

    # Determine per-category target
    samples_per_category = args.samples_per_category
    if args.n_chains is not None:
        # Legacy mode: split n_chains across categories
        samples_per_category = max(1, args.n_chains // len(categories))

    # Sharding
    rank_categories = [c for i, c in enumerate(categories) if i % args.world_size == args.rank]
    if not rank_categories:
        print(f"Rank {args.rank}: no categories assigned.")
        return

    if args.world_size > 1:
        args.output = args.output.replace(".json", f"_part{args.rank}.json")

    # Resume
    existing: List[dict] = []
    cat_counts: Counter = Counter()
    if os.path.exists(args.output):
        with open(args.output) as f:
            try:
                existing = json.load(f)
                cat_counts = Counter(c["category"] for c in existing)
                print(f"Resuming: {len(existing)} chains already saved, "
                      f"per-category counts: {dict(cat_counts)}")
            except json.JSONDecodeError:
                existing = []

    rank_categories = [c for c in rank_categories
                       if cat_counts.get(c["name"], 0) < samples_per_category]
    if not rank_categories:
        print("All categories complete.")
        return
    if args.prompt_template:
        gen_template = Path(args.prompt_template).read_text(encoding="utf-8")
    else:
        gen_template = _load_prompt("qa_gen_agent.txt")

    print(f"Generating {samples_per_category} samples x {len(rank_categories)} categories "
          f"(concurrency={args.concurrency}, model={model}, claude={claude_bin})")

    semaphore = asyncio.Semaphore(args.concurrency)
    results = list(existing)

    async def _process(cat: dict) -> List[dict]:
        async with semaphore:
            cat_remaining = samples_per_category - cat_counts.get(cat["name"], 0)
            print(f"\n-> Category: {cat['name']} ({cat_remaining} pairs remaining)")
            chains = await generate_for_category(
                category=cat,
                corpus_text_dir=corpus_text_dir,
                n_chains=cat_remaining,
                claude_bin=claude_bin,
                model=model,
                max_budget_usd=max_budget_usd,
                batch_size=args.batch_size,
                gen_template=gen_template,
                max_retries=args.max_retries,
                min_answer_len=args.min_answer_len,
                min_evidence_len=args.min_evidence_len,
                rng=rng,
                save_raw_runs=args.save_raw_runs,
                output_dir=output_dir,
                log_dir=args.log_dir,
            )
            return chains

    tasks = [_process(cat) for cat in rank_categories]
    for coro in asyncio.as_completed(tasks):
        cat_chains = await coro
        results.extend(cat_chains)
        _atomic_save(results, args.output)

    print(f"\nDone. {len(results)} total chains -> {args.output}")

    # Export grouped deliverable
    if args.export_deliverable:
        deliverable, errors = _build_grouped_deliverable(
            results, categories, samples_per_category,
        )
        if errors:
            print(f"\nDeliverable export warnings:")
            for err in errors:
                print(f"  - {err}")
        if deliverable["categories"]:
            deliverable_path = args.deliverable_output
            if not os.path.isabs(deliverable_path) and not os.path.dirname(deliverable_path):
                deliverable_path = os.path.join(output_dir, deliverable_path)
            _atomic_save_dict(deliverable, deliverable_path)
            print(f"Deliverable: {len(deliverable['categories'])} categories -> {deliverable_path}")

            # Export CSV
            csv_path = args.csv_output
            if not os.path.isabs(csv_path) and not os.path.dirname(csv_path):
                csv_path = os.path.join(output_dir, csv_path)
            _export_deliverable_csv(deliverable, csv_path)
            print(f"CSV: {sum(len(c['samples']) for c in deliverable['categories'])} rows -> {csv_path}")
        else:
            print("\nDeliverable export FAILED: no categories met the required sample count.")


if __name__ == "__main__":
    asyncio.run(main())

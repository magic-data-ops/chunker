#!/usr/bin/env python3
"""Launch the Q&A generation pipeline.

Each step runs `claude -p` directly (no server needed). Claude Code's native
Grep/Read/Glob tools operate on the corpus text directory.

Usage:
    python launch_qa_gen.py --step 2 --corpus_text_dir ./corpus_text --n_chains 15
    python launch_qa_gen.py --step all --corpus_text_dir ./corpus_text
    python launch_qa_gen.py --step 2 --world_size 4 --n_chains 2000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.resolve()


# ---------------------------------------------------------------------------
# Pipeline step runner
# ---------------------------------------------------------------------------


def _derive_multiturn_output(raw_output: str, explicit: str | None) -> str:
    """Derive the multiturn output path from raw_output or an explicit override."""
    if explicit:
        return explicit
    raw_path = Path(raw_output)
    stem = raw_path.stem
    if stem.endswith("_raw"):
        new_stem = stem[: -len("_raw")] + "_multiturn"
    else:
        new_stem = stem + "_multiturn"
    return str(raw_path.with_name(new_stem + raw_path.suffix))


def _run_step(
    step: str,
    args: argparse.Namespace,
    raw_output: str,
    validated_output: str,
) -> int:
    """Run a single pipeline step. Returns exit code."""
    python = sys.executable

    if step == "2":
        script = SCRIPT_DIR / "generate_qa_chains.py"
        cmd = [
            python, str(script),
            "--corpus_text_dir", args.corpus_text_dir,
            "--output", raw_output,
            "--samples-per-category", str(args.samples_per_category),
            "--batch_size", str(args.batch_size),
            "--concurrency", str(args.concurrency),
            "--claude-bin", args.claude_bin,
            "--model", args.model,
            "--max-budget-usd", str(args.max_budget_usd),
            "--deliverable-output", args.deliverable_output,
            "--csv-output", args.csv_output,
        ]
        if args.prompt_template:
            cmd += ["--prompt-template", args.prompt_template]
        if args.categories_cfg:
            cmd += ["--categories_cfg", args.categories_cfg]
        if args.save_raw_runs:
            cmd += ["--save-raw-runs"]
        if not args.export_deliverable:
            cmd += ["--no-export-deliverable"]
        if args.world_size > 1:
            return _run_multi_rank_step2(python, script, args, raw_output)

    elif step == "5":
        script = SCRIPT_DIR / "generate_multiturn.py"
        multiturn_output = _derive_multiturn_output(raw_output, args.multiturn_output)
        cmd = [
            python, str(script),
            "--input", raw_output,
            "--output", multiturn_output,
            "--concurrency", str(args.concurrency),
            "--model", args.model,
            "--max-budget-usd", str(args.max_budget_usd),
        ]
        if args.categories_cfg:
            cmd += ["--categories_cfg", args.categories_cfg]

    elif step == "3":
        script = SCRIPT_DIR / "contractor_polish.py"
        # Use multiturn output if it exists, unless --skip-multiturn was set
        multiturn_output = _derive_multiturn_output(raw_output, args.multiturn_output)
        if not args.skip_multiturn and os.path.exists(multiturn_output):
            polish_input = multiturn_output
        else:
            polish_input = raw_output
        cmd = [
            python, str(script),
            "--input", polish_input,
            "--output", validated_output,
            "--concurrency", str(args.concurrency),
        ]
        if args.categories_cfg:
            cmd += ["--categories_cfg", args.categories_cfg]

    elif step == "4":
        script = SCRIPT_DIR / "validate_qa_dataset.py"
        cmd = [
            python, str(script),
            "--input", validated_output,
            "--output", args.report_output,
        ]
    else:
        print(f"Unknown step: {step}")
        return 1

    print(f"\n=== Step {step}: {script.name} ===")
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    return result.returncode


def _run_multi_rank_step2(
    python: str,
    script: Path,
    args: argparse.Namespace,
    raw_output: str,
) -> int:
    """Shard step 2 across world_size subprocesses and merge partial outputs."""
    procs: List[subprocess.Popen] = []
    for rank in range(args.world_size):
        cmd = [
            python, str(script),
            "--corpus_text_dir", args.corpus_text_dir,
            "--output", raw_output,
            "--samples-per-category", str(args.samples_per_category),
            "--batch_size", str(args.batch_size),
            "--concurrency", str(args.concurrency),
            "--claude-bin", args.claude_bin,
            "--model", args.model,
            "--max-budget-usd", str(args.max_budget_usd),
            "--rank", str(rank),
            "--world_size", str(args.world_size),
            "--no-export-deliverable",
        ]
        if args.save_raw_runs:
            cmd += ["--save-raw-runs"]
        p = subprocess.Popen(cmd, cwd=str(PROJECT_DIR))
        procs.append(p)
        print(f"  Spawned rank {rank} (PID {p.pid})")

    for p in procs:
        p.wait()

    failed = [i for i, p in enumerate(procs) if p.returncode != 0]
    if failed:
        print(f"  Ranks failed: {failed}")
        return 1

    # Merge partial outputs
    all_chains = []
    for rank in range(args.world_size):
        part = raw_output.replace(".json", f"_part{rank}.json")
        if os.path.exists(part):
            with open(part) as f:
                all_chains.extend(json.load(f))
            os.remove(part)

    tmp = raw_output + ".tmp"
    with open(tmp, "w") as f:
        json.dump(all_chains, f, indent=2, ensure_ascii=False)
    if os.path.exists(raw_output):
        os.remove(raw_output)
    os.rename(tmp, raw_output)
    print(f"  Merged {len(all_chains)} chains -> {raw_output}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Q&A generation pipeline")
    parser.add_argument("--step", choices=["2", "3", "4", "5", "all"], default="2",
                        help="Pipeline step(s) to run (5 = multi-turn generation)")
    parser.add_argument("--corpus_text_dir", default="corpus_text",
                        help="Directory of .txt files for Claude Code to Grep/Read")
    parser.add_argument("--claude-bin", default="claude",
                        help="Path to claude binary (default: 'claude')")
    parser.add_argument("--model", default="sonnet",
                        choices=["sonnet", "opus", "haiku"],
                        help="Claude model to use (default: sonnet)")
    parser.add_argument("--max-budget-usd", type=float, default=1.00,
                        help="Cost cap per Claude Code invocation in USD")
    parser.add_argument("--samples-per-category", type=int, default=3,
                        help="Exact number of samples per category (default: 3)")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--save-raw-runs", action="store_true",
                        help="Save per-run raw JSONL output for debugging")
    parser.add_argument("--deliverable-output", default="qa_deliverable_grouped.json",
                        help="Path for grouped deliverable export")
    parser.add_argument("--csv-output", default="qa_deliverable.csv",
                        help="Path for CSV deliverable export")
    parser.add_argument("--prompt-template", default=None,
                        help="Path to agent prompt template")
    parser.add_argument("--categories_cfg", default=None,
                        help="Path to categories YAML config")
    parser.add_argument("--export-deliverable", action="store_true", default=True,
                        help="Export grouped deliverable (default: true)")
    parser.add_argument("--no-export-deliverable", action="store_false", dest="export_deliverable")
    parser.add_argument("--raw_output", default="qa_chains_raw.json")
    parser.add_argument("--validated_output", default="qa_chains_validated.json")
    parser.add_argument("--report_output", default="validation_report.json")
    parser.add_argument("--multiturn_output", default=None,
                        help="Path for multi-turn enriched chains (default: derived from raw_output)")
    parser.add_argument("--skip-multiturn", action="store_true",
                        help="Skip multi-turn generation step when running 'all'")
    args = parser.parse_args()

    if args.step == "all":
        steps = ["2", "5", "3", "4"] if not args.skip_multiturn else ["2", "3", "4"]
    else:
        steps = [args.step]
    exit_code = 0

    for step in steps:
        rc = _run_step(step, args, args.raw_output, args.validated_output)
        if rc != 0:
            print(f"Step {step} failed (exit {rc}). Stopping.")
            exit_code = rc
            break

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

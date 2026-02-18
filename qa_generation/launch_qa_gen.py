#!/usr/bin/env python3
"""Launch the Q&A generation pipeline.

Each step runs `opencode run` directly (no server needed). OpenCode's built-in
grep/read tools operate on the corpus text directory as if it were a project.

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


def _run_step(
    step: str,
    args: argparse.Namespace,
    raw_output: str,
    validated_output: str,
) -> int:
    """Run a single pipeline step. Returns exit code."""
    python = sys.executable

    if step == "2":
        script = SCRIPT_DIR / "2_generate_qa_chains.py"
        cmd = [
            python, str(script),
            "--corpus_text_dir", args.corpus_text_dir,
            "--output", raw_output,
            "--n_chains", str(args.n_chains),
            "--batch_size", str(args.batch_size),
            "--concurrency", str(args.concurrency),
        ]
        if args.opencode:
            cmd += ["--opencode", args.opencode]
        if args.world_size > 1:
            return _run_multi_rank_step2(python, script, args, raw_output)

    elif step == "3":
        script = SCRIPT_DIR / "3_contractor_polish.py"
        cmd = [
            python, str(script),
            "--input", raw_output,
            "--output", validated_output,
            "--concurrency", str(args.concurrency),
        ]

    elif step == "4":
        script = SCRIPT_DIR / "4_validate_qa_dataset.py"
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
            "--n_chains", str(args.n_chains),
            "--batch_size", str(args.batch_size),
            "--concurrency", str(args.concurrency),
            "--rank", str(rank),
            "--world_size", str(args.world_size),
        ]
        if args.opencode:
            cmd += ["--opencode", args.opencode]
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
    print(f"  Merged {len(all_chains)} chains → {raw_output}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Q&A generation pipeline")
    parser.add_argument("--step", choices=["2", "3", "4", "all"], default="2",
                        help="Pipeline step(s) to run")
    parser.add_argument("--corpus_text_dir", default="corpus_text",
                        help="Directory of .txt files for OpenCode to grep/read")
    parser.add_argument("--opencode", default=None,
                        help="Path to opencode binary (auto-detected if omitted)")
    parser.add_argument("--n_chains", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--raw_output", default="qa_chains_raw.json")
    parser.add_argument("--validated_output", default="qa_chains_validated.json")
    parser.add_argument("--report_output", default="validation_report.json")
    args = parser.parse_args()

    steps = ["2", "3", "4"] if args.step == "all" else [args.step]
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

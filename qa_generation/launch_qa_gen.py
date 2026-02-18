#!/usr/bin/env python3
"""Launch the Q&A generation pipeline backed by OpenCode as the LLM harness.

This script:
  1. Copies opencode.json into the corpus text directory
  2. Starts `opencode serve` from that directory (corpus = "the project")
  3. Waits until the HTTP server is accepting connections
  4. Runs the requested pipeline steps (2, 3, 4, or all) by spawning the
     Python scripts as subprocesses with OPENCODE_SERVER_URL injected
  5. Stops the OpenCode server when the pipeline finishes

Running OpenCode from the corpus directory means the agent's built-in grep
and read tools work on the corpus files naturally, using relative paths — no
absolute path plumbing needed in prompts.

Works with the officially installed opencode binary (no source build needed).
For multi-GPU generation use --world_size with step 2; step results are
merged automatically.

Usage:
    python launch_qa_gen.py --step 2 --corpus_text_dir ./corpus_text --n_chains 200
    python launch_qa_gen.py --step all --corpus_text_dir ./corpus_text
    python launch_qa_gen.py --step 2 --world_size 4 --n_chains 2000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.resolve()
DEFAULT_PORT = 4096

# Locate opencode binary: prefer $PATH, then common install locations
def _find_opencode() -> str:
    import shutil
    if shutil.which("opencode"):
        return "opencode"
    candidates = [
        Path.home() / ".opencode" / "bin" / "opencode",
        Path.home() / ".local" / "bin" / "opencode",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "opencode"  # will fail loudly at launch time

DEFAULT_OPENCODE_BIN = _find_opencode()


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


async def _wait_for_server(url: str, retries: int = 40, interval: float = 0.75) -> None:
    """Poll the server until it responds or we exhaust retries."""
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{url}/doc")
                if resp.status_code < 500:
                    return
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError):
            pass
        await asyncio.sleep(interval)
    raise TimeoutError(f"OpenCode server at {url} did not become ready after {retries * interval:.0f}s")


def _start_opencode_server(port: int, opencode_bin: str, cwd: str) -> subprocess.Popen:
    """Start `opencode serve` from cwd (the corpus directory) as a background process."""
    cmd = [
        opencode_bin,
        "serve",
        "--port", str(port),
        "--hostname", "127.0.0.1",
    ]
    print(f"Starting OpenCode server (cwd={cwd}): {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def _stop_opencode_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("OpenCode server stopped.")


def _prepare_corpus_dir(corpus_text_dir: str) -> None:
    """Copy opencode.json into corpus_text_dir so OpenCode finds its config there."""
    src = PROJECT_DIR / "opencode.json"
    dst = Path(corpus_text_dir) / "opencode.json"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied opencode.json → {dst}")
    else:
        print(f"WARNING: {src} not found — OpenCode may not find agent configs")


# ---------------------------------------------------------------------------
# Pipeline step runner
# ---------------------------------------------------------------------------


def _run_step(
    step: str,
    server_url: str,
    corpus_text_dir: str,
    args: argparse.Namespace,
    raw_output: str,
    validated_output: str,
) -> int:
    """Run a single pipeline step. Returns exit code."""
    python = sys.executable
    env = {**os.environ, "OPENCODE_SERVER_URL": server_url}

    if step == "2":
        script = SCRIPT_DIR / "2_generate_qa_chains.py"
        cmd = [
            python, str(script),
            "--corpus_text_dir", corpus_text_dir,
            "--output", raw_output,
            "--n_chains", str(args.n_chains),
            "--concurrency", str(args.concurrency),
        ]
        if args.world_size > 1:
            return _run_multi_rank_step2(python, script, corpus_text_dir, args, raw_output, env)

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
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_DIR))
    return result.returncode


def _run_multi_rank_step2(
    python: str,
    script: Path,
    corpus_text_dir: str,
    args: argparse.Namespace,
    raw_output: str,
    env: dict,
) -> int:
    """Shard step 2 across world_size subprocesses and merge partial outputs."""
    procs: List[subprocess.Popen] = []
    for rank in range(args.world_size):
        cmd = [
            python, str(script),
            "--corpus_text_dir", corpus_text_dir,
            "--output", raw_output,
            "--n_chains", str(args.n_chains),
            "--concurrency", str(args.concurrency),
            "--rank", str(rank),
            "--world_size", str(args.world_size),
        ]
        p = subprocess.Popen(cmd, env=env, cwd=str(PROJECT_DIR))
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


async def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Q&A generation pipeline via OpenCode")
    parser.add_argument("--step", choices=["2", "3", "4", "all"], default="2",
                        help="Pipeline step(s) to run")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="Port for the OpenCode server")
    parser.add_argument("--opencode", default=DEFAULT_OPENCODE_BIN,
                        help="Path to the opencode binary")
    parser.add_argument("--corpus_text_dir", default="corpus_text",
                        help="Directory of .txt files — OpenCode runs from here")
    parser.add_argument("--n_chains", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--raw_output", default="qa_chains_raw.json")
    parser.add_argument("--validated_output", default="qa_chains_validated.json")
    parser.add_argument("--report_output", default="validation_report.json")
    args = parser.parse_args()

    corpus_text_dir = str(Path(args.corpus_text_dir).resolve())
    server_url = f"http://127.0.0.1:{args.port}"
    steps = ["2", "3", "4"] if args.step == "all" else [args.step]

    # Copy opencode.json into the corpus dir so the server picks it up
    _prepare_corpus_dir(corpus_text_dir)

    opencode_proc: Optional[subprocess.Popen] = None
    exit_code = 0

    try:
        opencode_proc = _start_opencode_server(args.port, args.opencode, corpus_text_dir)
        print(f"Waiting for OpenCode server at {server_url}...")
        await _wait_for_server(server_url)
        print(f"OpenCode server ready at {server_url}")

        for step in steps:
            rc = _run_step(step, server_url, corpus_text_dir, args, args.raw_output, args.validated_output)
            if rc != 0:
                print(f"Step {step} failed (exit {rc}). Stopping.")
                exit_code = rc
                break

    except TimeoutError as e:
        print(f"ERROR: {e}")
        exit_code = 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        exit_code = 130
    finally:
        if opencode_proc:
            _stop_opencode_server(opencode_proc)

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())

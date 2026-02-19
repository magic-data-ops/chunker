"""QAAgent: runs Claude Code CLI as the LLM harness for pipeline steps.

The interface is unchanged — callers just do:

    agent = QAAgent("contractor_validator")
    reply = await agent.generate(prompt)

Under the hood every call runs `claude -p` as a subprocess with the prompt.
No server needed.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("qa_generation.log", encoding="utf-8")],
)
logger = logging.getLogger("QAAgent")


class QAAgent:
    """Async LLM agent backed by Claude Code CLI.

    The agent name is used for logging only. All agents use the same
    `claude -p` subprocess interface.
    """

    def __init__(
        self,
        name: str,
        claude_bin: str = "claude",
        model: str = "sonnet",
        max_budget_usd: float = 0.10,
    ):
        self.name = name
        self.claude_bin = claude_bin
        self.model = model
        self.max_budget_usd = max_budget_usd

    async def generate(self, prompt: str, timeout: float = 300.0) -> str:
        """Send prompt to Claude Code CLI and return the reply text."""
        cmd = [
            self.claude_bin, "-p", prompt,
            "--output-format", "json",
            "--dangerously-skip-permissions",
            "--model", self.model,
            "--max-budget-usd", str(self.max_budget_usd),
        ]

        # Filter out CLAUDECODE env var to avoid nested-session error
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        try:
            logger.info(f"REQ [{self.name}] via Claude Code CLI (model={self.model})")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            raw = stdout_bytes.decode(errors="replace").strip()

            # --output-format json returns a single JSON object with a "result" field
            import json
            try:
                data = json.loads(raw)
                reply = data.get("result", raw)
            except json.JSONDecodeError:
                reply = raw

            logger.info(f"RES [{self.name}] len={len(reply)}")
            return reply
        except asyncio.TimeoutError:
            logger.error(f"TIMEOUT [{self.name}]")
            return "[ERROR] timeout"
        except Exception as e:
            logger.error(f"ERR [{self.name}] {e}")
            return f"[ERROR] {e}"

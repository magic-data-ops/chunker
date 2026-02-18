"""QAAgent: wraps OpenCodeClient so pipeline scripts call OpenCode as their LLM harness.

The interface is unchanged from the original — callers just do:

    agent = QAAgent("qa_proposer")
    reply = await agent.generate(prompt)

Under the hood every call:
  1. Creates a fresh OpenCode session scoped to the current project
  2. Sends the rendered prompt as a user message to the named agent
  3. Returns the assistant's text
  4. Deletes the session (cleanup)

OpenCode picks up the agent's system prompt, model, and temperature from
opencode.json in the project root (~/magic/chunker/opencode.json).
"""

from __future__ import annotations

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

# Resolve client lazily so the import doesn't fail if OPENCODE_SERVER_URL is unset
_client: Optional[object] = None


def _get_client():
    global _client
    if _client is None:
        from utils.opencode_client import OpenCodeClient
        _client = OpenCodeClient()
    return _client


class QAAgent:
    """Async LLM agent backed by the OpenCode HTTP session API.

    The agent name must match a key in opencode.json → "agent" section.
    Valid names: "qa_proposer", "hop_answerer", "contractor_validator".
    """

    def __init__(self, name: str):
        self.name = name

    async def generate(self, prompt: str) -> str:
        """Send prompt to the named OpenCode agent and return the reply."""
        client = _get_client()
        try:
            logger.info(f"REQ [{self.name}] via OpenCode session API")
            reply = await client.complete(self.name, prompt)
            logger.info(f"RES [{self.name}] len={len(reply)}")
            return reply
        except Exception as e:
            logger.error(f"ERR [{self.name}] {e}")
            return f"[ERROR] {e}"

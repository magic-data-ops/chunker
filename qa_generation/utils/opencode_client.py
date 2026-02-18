"""OpenCodeClient: async HTTP client for the OpenCode session API.

The OpenCode server is started externally (via `opencode serve` or
`opencode qa-gen`) and its URL is advertised via the OPENCODE_SERVER_URL
environment variable.  The project directory is passed as a query parameter
so the server uses the correct opencode.json config (and therefore the right
agent definitions).

Usage:
    client = OpenCodeClient()
    text = await client.complete("qa_proposer", rendered_prompt)
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

import httpx


class OpenCodeClient:
    """Thin async wrapper around the OpenCode HTTP session API.

    For each LLM call we:
      1. POST /session        → get session_id
      2. POST /session/:id/message  → get assistant text
      3. DELETE /session/:id  → clean up

    The `directory` query param tells the server which project's opencode.json
    to use for agent definitions.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        directory: Optional[str] = None,
        timeout: float = 180.0,
    ):
        self.base_url = (base_url or os.getenv("OPENCODE_SERVER_URL", "http://localhost:4096")).rstrip("/")
        self.directory = directory or os.getenv("OPENCODE_PROJECT_DIR", os.getcwd())
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Low-level primitives
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        return {"x-opencode-directory": self.directory, "Content-Type": "application/json"}

    def _params(self) -> dict:
        return {"directory": self.directory}

    async def create_session(self, title: str = "qa_gen") -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/session",
                json={"title": title},
                headers=self._headers(),
                params=self._params(),
            )
            resp.raise_for_status()
            data = resp.json()
            return data["id"]

    async def send_prompt(self, session_id: str, agent: str, text: str) -> str:
        """Send a user message to a session and return the assistant's text."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/session/{session_id}/message",
                json={
                    "parts": [{"type": "text", "text": text}],
                    "agent": agent,
                },
                headers=self._headers(),
                params=self._params(),
            )
            resp.raise_for_status()

            # The response is application/json — one JSON object written after the LLM completes.
            # Structure: { "info": {...}, "parts": [{ "type": "text", "text": "..." }, ...] }
            raw = resp.text.strip()
            # Handle streaming: if multiple JSON objects are concatenated, take the last complete one
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Try last line (some streaming responses end with the final object on last line)
                for line in reversed(raw.splitlines()):
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            data = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue
                else:
                    return f"[ERROR] Could not parse response: {raw[:200]}"

            parts = data.get("parts", [])
            text_parts = [p.get("text", "") for p in parts if p.get("type") == "text"]
            return " ".join(t for t in text_parts if t).strip()

    async def delete_session(self, session_id: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.delete(
                    f"{self.base_url}/session/{session_id}",
                    headers=self._headers(),
                    params=self._params(),
                )
        except Exception:
            pass  # best-effort cleanup

    # ------------------------------------------------------------------
    # High-level convenience
    # ------------------------------------------------------------------

    async def complete(self, agent: str, prompt: str) -> str:
        """Create a session, send a prompt, return the reply, then delete the session."""
        session_id = await self.create_session(title=f"qa_{agent}")
        try:
            return await self.send_prompt(session_id, agent, prompt)
        finally:
            await self.delete_session(session_id)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def wait_until_ready(self, retries: int = 30, interval: float = 1.0) -> None:
        """Poll the server until it responds, up to retries * interval seconds."""
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{self.base_url}/doc")
                    if resp.status_code < 500:
                        return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            await asyncio.sleep(interval)
        raise TimeoutError(
            f"OpenCode server at {self.base_url} did not become ready after {retries * interval:.0f}s"
        )

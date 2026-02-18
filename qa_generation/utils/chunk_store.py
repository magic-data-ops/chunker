"""ChunkStore: load corpus_chunks.json and look up chunks by id."""

from __future__ import annotations

import json
from typing import Dict, List, Optional


class ChunkStore:
    """In-memory lookup table for chunk records.

    Expects the corpus produced by ``1_build_corpus_index.py``:
    a JSON array of objects, each with at minimum a ``chunk_id`` key.
    """

    def __init__(self, chunks_path: str):
        with open(chunks_path, "r", encoding="utf-8") as f:
            raw: List[dict] = json.load(f)
        self._store: Dict[str, dict] = {c["chunk_id"]: c for c in raw}
        self._ids: List[str] = list(self._store.keys())

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, chunk_id: str) -> Optional[dict]:
        """Return chunk record or None if not found."""
        return self._store.get(chunk_id)

    def get_or_raise(self, chunk_id: str) -> dict:
        chunk = self.get(chunk_id)
        if chunk is None:
            raise KeyError(f"chunk_id '{chunk_id}' not found in ChunkStore")
        return chunk

    def all_ids(self) -> List[str]:
        return list(self._ids)

    def all_chunks(self) -> List[dict]:
        return list(self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._store

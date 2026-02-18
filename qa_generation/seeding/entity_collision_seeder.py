"""EntityCollisionSeeder: cycle through NER-detected entity collision pairs."""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING, List, Tuple

from seeding.base_seeder import BaseSeeder

if TYPE_CHECKING:
    from utils.chunk_store import ChunkStore


class EntityCollisionSeeder(BaseSeeder):
    """Seeds from chunks that contain ambiguous entity mentions.

    Loads ``entity_collision_index.json`` and flattens it into a shuffled
    pool of ``(entity_name, chunk_id)`` pairs.  A round-robin pointer
    walks through the pool so every collision pair is eventually visited.

    The entity name is injected into the returned chunk dict under
    ``_seed_entity`` for use by the proposer prompt.
    """

    def __init__(self, chunk_store: "ChunkStore", collision_index_path: str, seed: int | None = None):
        super().__init__(chunk_store)
        self._pointer = 0
        self._seed_pool: List[Tuple[str, str]] = []
        self.prepare(collision_index_path, seed)

    def prepare(self, collision_index_path: str, seed: int | None = None) -> None:
        """Flatten the collision index into a shuffled seed pool."""
        with open(collision_index_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        collisions: dict = data.get("collisions", {})
        pool: List[Tuple[str, str]] = []

        for entity_name, info in collisions.items():
            for chunk_entry in info.get("chunks", []):
                chunk_id = chunk_entry["chunk_id"]
                pool.append((entity_name, chunk_id))

        rng = random.Random(seed)
        rng.shuffle(pool)
        self._seed_pool = pool

        if not self._seed_pool:
            raise ValueError(
                f"Entity collision index at '{collision_index_path}' produced an empty seed pool. "
                "Run 1_build_corpus_index.py with NER enabled first."
            )

    def get_seed_chunk(self) -> dict:
        if not self._seed_pool:
            raise RuntimeError("Seed pool is empty — call prepare() first.")

        entity_name, chunk_id = self._seed_pool[self._pointer % len(self._seed_pool)]
        self._pointer += 1

        chunk = self.chunk_store.get_or_raise(chunk_id).copy()
        chunk["_seed_entity"] = entity_name
        return chunk

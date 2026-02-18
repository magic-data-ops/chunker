"""RandomSeeder: uniformly random chunk selection."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from seeding.base_seeder import BaseSeeder

if TYPE_CHECKING:
    from utils.chunk_store import ChunkStore


class RandomSeeder(BaseSeeder):
    """Returns a uniformly random chunk from the store each call."""

    def __init__(self, chunk_store: "ChunkStore", seed: int | None = None):
        super().__init__(chunk_store)
        self._rng = random.Random(seed)

    def get_seed_chunk(self) -> dict:
        chunk_id = self._rng.choice(self.chunk_store.all_ids())
        return self.chunk_store.get_or_raise(chunk_id)

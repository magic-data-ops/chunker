"""Abstract BaseSeeder interface for all seeding strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qa_generation.utils.chunk_store import ChunkStore


class BaseSeeder(ABC):
    """All seeders must implement ``get_seed_chunk``."""

    def __init__(self, chunk_store: "ChunkStore"):
        self.chunk_store = chunk_store

    @abstractmethod
    def get_seed_chunk(self) -> dict:
        """Return a chunk record to start a new hop chain from."""
        ...

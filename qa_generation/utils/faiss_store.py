"""FAISSStore: add, search, save/load a flat inner-product index."""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np


class FAISSStore:
    """Thin wrapper around faiss.IndexFlatIP.

    All vectors are L2-normalised before insertion so that inner product
    equals cosine similarity.
    """

    def __init__(self, dim: int, use_ivf: bool = False, nlist: int = 100):
        import faiss  # local import — optional dependency

        self.dim = dim
        self.use_ivf = use_ivf
        self._id_map: List[str] = []  # position → chunk_id

        if use_ivf:
            quantiser = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            self._index = faiss.IndexFlatIP(dim)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add(self, chunk_id: str, vector: np.ndarray) -> None:
        """Normalise and add a single vector."""
        vec = self._normalise(vector).reshape(1, -1).astype(np.float32)
        if self.use_ivf and not self._index.is_trained:
            # IVF requires training; caller is responsible for calling train() first
            raise RuntimeError("IVF index is not trained. Call train() before add().")
        self._index.add(vec)
        self._id_map.append(chunk_id)

    def add_batch(self, chunk_ids: List[str], vectors: np.ndarray) -> None:
        """Normalise and add a batch of vectors (N × dim)."""
        vecs = self._normalise_batch(vectors).astype(np.float32)
        if self.use_ivf and not self._index.is_trained:
            raise RuntimeError("IVF index is not trained. Call train() before add_batch().")
        self._index.add(vecs)
        self._id_map.extend(chunk_ids)

    def train(self, vectors: np.ndarray) -> None:
        """Train IVF index (no-op for flat index)."""
        if self.use_ivf:
            vecs = self._normalise_batch(vectors).astype(np.float32)
            self._index.train(vecs)

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Return up to *k* (chunk_id, score) pairs, best-first."""
        q = self._normalise(query_vector).reshape(1, -1).astype(np.float32)
        k = min(k, len(self._id_map))
        if k == 0:
            return []
        scores, indices = self._index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_path: str, id_map_path: str) -> None:
        """Save FAISS index and id-map JSON side-car."""
        import faiss

        faiss.write_index(self._index, index_path)
        with open(id_map_path, "w", encoding="utf-8") as f:
            json.dump(self._id_map, f)

    @classmethod
    def load(cls, index_path: str, id_map_path: str) -> "FAISSStore":
        """Load a previously saved FAISSStore."""
        import faiss

        index = faiss.read_index(index_path)
        with open(id_map_path, "r", encoding="utf-8") as f:
            id_map = json.load(f)

        store = cls.__new__(cls)
        store._index = index
        store._id_map = id_map
        store.dim = index.d
        store.use_ivf = hasattr(index, "nlist")
        return store

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    @staticmethod
    def _normalise_batch(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms

    def __len__(self) -> int:
        return len(self._id_map)

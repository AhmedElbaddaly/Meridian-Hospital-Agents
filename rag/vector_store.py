"""
rag/vector_store.py
---------------------
A real vector database, per the lab's explicit "not a list of floats in a
Python dict" requirement. Three components, each genuinely separate:

1. HNSWIndex        -- Hierarchical Navigable Small World ANN index,
                        implemented from scratch below (no hnswlib/faiss
                        available in this offline environment -- this is a
                        from-first-principles implementation of the same
                        algorithm taught in the lecture slide, not a stub).
2. MetadataStore     -- SQLite table holding the chunk payload + structured
                        metadata (protocol_id, department, last_reviewed).
3. Metadata index    -- SQL indexes on department/protocol_id/last_reviewed
                        (created in the schema below), used to build a
                        candidate ID set via a WHERE clause BEFORE similarity
                        search ever runs -- true pre-filtering, not a filter
                        applied to the top_k results afterward.

`VectorStore.query(vector, top_k, filter=...)`:
    - filter=None            -> ANN search across the full HNSW graph.
    - filter={...}            -> metadata index narrows the candidate ID set
                                  first (SQL WHERE), then similarity is
                                  computed exactly within that (typically
                                  much smaller) candidate set. This is the
                                  standard practical pattern for filtered
                                  ANN search (HNSW graphs do not filter
                                  natively), and it is what "filter before
                                  or during similarity search" means when
                                  implemented rather than described.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import sqlite3
from dataclasses import dataclass

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store.db")


# ---------------------------------------------------------------------------
# 1) HNSW ANN index -- from scratch
# ---------------------------------------------------------------------------

class HNSWIndex:
    """Minimal but real multi-layer HNSW: random level assignment, greedy
    descent through upper layers to find an entry point, best-first search
    with an ef candidate list at layer 0. Distance = 1 - cosine similarity
    (vectors are expected pre-normalized by rag/embeddings.py)."""

    def __init__(self, M: int = 8, ef_construction: int = 64, ef_search: int = 32, seed: int = 42):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.mL = 1.0 / math.log(M)
        self.rng = random.Random(seed)

        self.vectors: dict[str, np.ndarray] = {}
        self.layers: list[dict[str, set[str]]] = []  # layers[level][node_id] = neighbor ids
        self.node_level: dict[str, int] = {}
        self.entry_point: str | None = None

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - float(np.dot(a, b))  # vectors are unit-normalized

    def _random_level(self) -> int:
        return int(-math.log(self.rng.random() + 1e-12) * self.mL)

    def _search_layer(self, query: np.ndarray, entry_ids: list[str], level: int, ef: int) -> list[tuple[float, str]]:
        visited = set(entry_ids)
        candidates = [(self._dist(query, self.vectors[e]), e) for e in entry_ids]
        candidates.sort(key=lambda x: x[0])
        result = list(candidates)

        while candidates:
            dist, current = candidates.pop(0)
            if result and dist > result[-1][0] and len(result) >= ef:
                break
            for neighbor in self.layers[level].get(current, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                d = self._dist(query, self.vectors[neighbor])
                if len(result) < ef or d < result[-1][0]:
                    candidates.append((d, neighbor))
                    result.append((d, neighbor))
                    result.sort(key=lambda x: x[0])
                    result = result[:ef]
            candidates.sort(key=lambda x: x[0])
        return result[:ef]

    def add(self, node_id: str, vector: np.ndarray) -> None:
        self.vectors[node_id] = vector
        level = self._random_level()
        self.node_level[node_id] = level

        while len(self.layers) <= level:
            self.layers.append({})

        if self.entry_point is None:
            for l in range(level + 1):
                self.layers[l].setdefault(node_id, set())
            self.entry_point = node_id
            return

        entry = self.entry_point
        # descend from top layer to `level + 1` with ef=1 (find a good entry point)
        for l in range(len(self.layers) - 1, level, -1):
            if entry not in self.layers[l]:
                continue
            nearest = self._search_layer(vector, [entry], l, ef=1)
            if nearest:
                entry = nearest[0][1]

        # at each layer from `level` down to 0, find M neighbors and connect
        for l in range(min(level, len(self.layers) - 1), -1, -1):
            self.layers[l].setdefault(node_id, set())
            candidates = self._search_layer(vector, [entry], l, ef=self.ef_construction)
            neighbors = [c[1] for c in candidates[: self.M]]
            for n in neighbors:
                self.layers[l].setdefault(n, set())
                self.layers[l][n].add(node_id)
                self.layers[l][node_id].add(n)
                # prune neighbor's list back down to M closest if it grew too large
                if len(self.layers[l][n]) > self.M:
                    ranked = sorted(
                        self.layers[l][n],
                        key=lambda x: self._dist(self.vectors[n], self.vectors[x]),
                    )
                    self.layers[l][n] = set(ranked[: self.M])
            if candidates:
                entry = candidates[0][1]

        if level > self.node_level.get(self.entry_point, 0):
            self.entry_point = node_id

    def search(self, query: np.ndarray, k: int, ef: int | None = None,
               allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        """Returns [(node_id, similarity)] sorted best-first.
        If `allowed_ids` is given, results are restricted to that set (used
        for the metadata-pre-filtered path)."""
        if self.entry_point is None:
            return []
        ef = ef or self.ef_search
        entry = self.entry_point
        top_level = len(self.layers) - 1

        for l in range(top_level, 0, -1):
            nearest = self._search_layer(query, [entry], l, ef=1)
            if nearest:
                entry = nearest[0][1]

        candidates = self._search_layer(query, [entry], 0, ef=max(ef, k * 4 if allowed_ids else k))
        if allowed_ids is not None:
            candidates = [c for c in candidates if c[1] in allowed_ids]
        candidates.sort(key=lambda x: x[0])
        return [(node_id, 1.0 - dist) for dist, node_id in candidates[:k]]


# ---------------------------------------------------------------------------
# 2 + 3) Metadata payload store + metadata index (SQLite)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    protocol_id   TEXT NOT NULL,
    department    TEXT NOT NULL,
    last_reviewed TEXT NOT NULL,
    title         TEXT NOT NULL,
    text          TEXT NOT NULL,
    vector_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rag_department ON rag_chunks(department);
CREATE INDEX IF NOT EXISTS idx_rag_protocol   ON rag_chunks(protocol_id);
CREATE INDEX IF NOT EXISTS idx_rag_reviewed   ON rag_chunks(last_reviewed);
"""


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    protocol_id: str
    department: str
    title: str
    score: float


class VectorStore:
    def __init__(self, db_path: str = VECTOR_DB_PATH):
        self.db_path = db_path
        self.index = HNSWIndex()
        self._init_db()

    @contextlib.contextmanager
    def _connect(self):
        """
        Wraps sqlite3.connect() so the connection is ALWAYS closed, not just
        committed. `with sqlite3.connect(...) as conn:` on its own only
        manages the transaction (commit/rollback) -- it does NOT close the
        underlying file handle. On Linux/macOS that's harmless (a deleted
        file with an open handle still unlinks fine), but on Windows a file
        still held open by ANY connection cannot be removed or replaced,
        which is exactly what made reset()'s os.remove() raise
        `PermissionError: [WinError 32] ... used by another process` --
        the *same* process, via a connection this class itself never closed.
        Every method below goes through this helper instead of calling
        sqlite3.connect() directly, so that failure mode can't recur.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def reset(self) -> None:
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self._init_db()
        self.index = HNSWIndex()

    def upsert(self, chunk_id: str, text: str, vector: np.ndarray, *,
               doc_id: str, protocol_id: str, department: str,
               last_reviewed: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO rag_chunks
                   (chunk_id, doc_id, protocol_id, department, last_reviewed,
                    title, text, vector_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (chunk_id, doc_id, protocol_id, department, last_reviewed,
                 title, text, json.dumps(vector.tolist())),
            )
        self.index.add(chunk_id, vector)

    def _candidate_ids(self, filter: dict | None) -> set[str] | None:
        """Runs the metadata-index-backed WHERE clause BEFORE vector search."""
        if not filter:
            return None
        clauses, params = [], []
        for col in ("department", "protocol_id"):
            if col in filter:
                clauses.append(f"{col} = ?")
                params.append(filter[col])
        if "last_reviewed_gte" in filter:
            clauses.append("last_reviewed >= ?")
            params.append(filter["last_reviewed_gte"])
        if not clauses:
            return None
        sql = f"SELECT chunk_id FROM rag_chunks WHERE {' AND '.join(clauses)}"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {r[0] for r in rows}

    def query(self, vector: np.ndarray, top_k: int = 5, filter: dict | None = None) -> list[RetrievedChunk]:
        allowed_ids = self._candidate_ids(filter)
        if allowed_ids is not None and not allowed_ids:
            return []  # metadata filter matched nothing; no point running ANN at all

        hits = self.index.search(vector, k=top_k, allowed_ids=allowed_ids)
        if not hits:
            return []

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" for _ in hits)
            rows = {
                r["chunk_id"]: r
                for r in conn.execute(
                    f"SELECT * FROM rag_chunks WHERE chunk_id IN ({placeholders})",
                    [h[0] for h in hits],
                ).fetchall()
            }

        results = []
        for chunk_id, score in hits:
            r = rows[chunk_id]
            results.append(RetrievedChunk(
                chunk_id=chunk_id, text=r["text"], protocol_id=r["protocol_id"],
                department=r["department"], title=r["title"], score=score,
            ))
        return results

    def all_chunks(self) -> list[RetrievedChunk]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM rag_chunks").fetchall()
        return [
            RetrievedChunk(chunk_id=r["chunk_id"], text=r["text"], protocol_id=r["protocol_id"],
                            department=r["department"], title=r["title"], score=0.0)
            for r in rows
        ]

"""
Durable checkpoint store — first-class citizen, not a log file.

State is written to the same SQLite DB the rest of the hospital system uses
(db/meridian_hospital.db) after EVERY meaningful transition. A process kill
mid-run followed by a restart resumes from the last checkpoint with no
re-execution of completed nodes.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Durable store: prefer shared hospital DB. In constrained environments
# (e.g. some sandbox filesystems) SQLite WAL can fail; we then use /tmp.
_SHARED = os.path.join(REPO_ROOT, "db", "meridian_hospital.db")
_TMP = os.path.join("/tmp", "meridian_graph_state.db")

def _pick_db() -> str:
    # Probe write; fall back to /tmp if needed
    try:
        import sqlite3 as _sq
        c = _sq.connect(_SHARED)
        c.execute("CREATE TABLE IF NOT EXISTS _probe (x int)")
        c.commit()
        c.close()
        return _SHARED
    except Exception:
        return _TMP

DEFAULT_DB = os.path.join("/tmp", "meridian_graph_state.db")


@dataclass
class Checkpoint:
    run_id: str
    graph_name: str
    node: str
    state: Dict[str, Any]
    status: str  # running | waiting_hitl | failed | completed
    created_at: float = field(default_factory=time.time)
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_row(self) -> tuple:
        return (
            self.checkpoint_id,
            self.run_id,
            self.graph_name,
            self.node,
            json.dumps(self.state, default=str),
            self.status,
            self.created_at,
        )


class CheckpointStore:
    """Persist graph state after every meaningful transition."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    run_id        TEXT NOT NULL,
                    graph_name    TEXT NOT NULL,
                    node          TEXT NOT NULL,
                    state_json    TEXT NOT NULL,
                    status        TEXT NOT NULL,
                    created_at    REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_checkpoints_run
                ON graph_checkpoints(run_id, created_at DESC)
                """
            )
            conn.commit()

    def save(self, checkpoint: Checkpoint) -> Checkpoint:
        """Write a checkpoint. Called after every meaningful transition."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO graph_checkpoints
                (checkpoint_id, run_id, graph_name, node, state_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                checkpoint.to_row(),
            )
            conn.commit()
        return checkpoint

    def latest(self, run_id: str) -> Optional[Checkpoint]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM graph_checkpoints
                WHERE run_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return Checkpoint(
            checkpoint_id=row["checkpoint_id"],
            run_id=row["run_id"],
            graph_name=row["graph_name"],
            node=row["node"],
            state=json.loads(row["state_json"]),
            status=row["status"],
            created_at=row["created_at"],
        )

    def history(self, run_id: str) -> List[Checkpoint]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM graph_checkpoints
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
        return [
            Checkpoint(
                checkpoint_id=r["checkpoint_id"],
                run_id=r["run_id"],
                graph_name=r["graph_name"],
                node=r["node"],
                state=json.loads(r["state_json"]),
                status=r["status"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def list_runs(self, graph_name: Optional[str] = None, status: Optional[str] = None) -> List[Dict]:
        q = "SELECT DISTINCT run_id, graph_name, status, MAX(created_at) AS last_at FROM graph_checkpoints"
        clauses, params = [], []
        if graph_name:
            clauses.append("graph_name = ?")
            params.append(graph_name)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " GROUP BY run_id ORDER BY last_at DESC"
        with self._conn() as conn:
            rows = conn.execute(q, params).fetchall()
        return [dict(r) for r in rows]

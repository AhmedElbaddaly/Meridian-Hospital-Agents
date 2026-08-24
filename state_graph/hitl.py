"""
Human-in-the-loop (HITL) escalation.

HITL is an *expected* pause: the agent is not allowed to decide alone
(amount above threshold, action that contradicts policy, confidence below bar).

When a HITL condition fires:
  1. Graph pauses
  2. Full state is checkpointed with status=waiting_hitl
  3. A task is opened for a real admin (surfaced on the platform UI)
  4. Graph resumes ONLY after the admin acts through the platform
  5. Resumed run picks up the admin's decision

This is intentionally distinct from the failure-ticket path.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from state_graph.checkpoint import DEFAULT_DB

@dataclass
class HITLTask:
    task_id: str
    run_id: str
    graph_name: str
    node: str
    reason: str
    state_snapshot: Dict[str, Any]
    status: str = "pending"  # pending | approved | rejected | modified
    admin_decision: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None


def _conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(db_path: Optional[str] = None) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hitl_tasks (
                task_id        TEXT PRIMARY KEY,
                run_id         TEXT NOT NULL,
                graph_name     TEXT NOT NULL,
                node           TEXT NOT NULL,
                reason         TEXT NOT NULL,
                state_json     TEXT NOT NULL,
                status         TEXT NOT NULL,
                admin_decision TEXT,
                created_at     REAL NOT NULL,
                resolved_at    REAL
            )
            """
        )
        conn.commit()


def open_hitl_task(
    run_id: str,
    graph_name: str,
    node: str,
    reason: str,
    state_snapshot: Dict[str, Any],
    db_path: Optional[str] = None,
) -> HITLTask:
    """Create a pending HITL task that the platform admin UI will surface."""
    _ensure_schema(db_path)
    task = HITLTask(
        task_id=str(uuid.uuid4()),
        run_id=run_id,
        graph_name=graph_name,
        node=node,
        reason=reason,
        state_snapshot=state_snapshot,
    )
    with _conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO hitl_tasks
            (task_id, run_id, graph_name, node, reason, state_json, status, admin_decision, created_at, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task.task_id,
                task.run_id,
                task.graph_name,
                task.node,
                task.reason,
                json.dumps(task.state_snapshot, default=str),
                task.status,
                None,
                task.created_at,
                None,
            ),
        )
        conn.commit()
    return task


def resolve_hitl_task(
    task_id: str,
    decision: Dict[str, Any],
    status: str = "approved",
    db_path: Optional[str] = None,
) -> HITLTask:
    """Admin acts through the platform; this records the decision and unblocks the graph."""
    _ensure_schema(db_path)
    now = time.time()
    with _conn(db_path) as conn:
        conn.execute(
            """
            UPDATE hitl_tasks
            SET status = ?, admin_decision = ?, resolved_at = ?
            WHERE task_id = ?
            """,
            (status, json.dumps(decision, default=str), now, task_id),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM hitl_tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
    if not row:
        raise ValueError(f"HITL task {task_id} not found")
    return HITLTask(
        task_id=row["task_id"],
        run_id=row["run_id"],
        graph_name=row["graph_name"],
        node=row["node"],
        reason=row["reason"],
        state_snapshot=json.loads(row["state_json"]),
        status=row["status"],
        admin_decision=json.loads(row["admin_decision"]) if row["admin_decision"] else None,
        created_at=row["created_at"],
        resolved_at=row["resolved_at"],
    )


def list_pending_hitl(db_path: Optional[str] = None) -> List[HITLTask]:
    _ensure_schema(db_path)
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM hitl_tasks WHERE status = 'pending' ORDER BY created_at ASC"
        ).fetchall()
    return [
        HITLTask(
            task_id=r["task_id"],
            run_id=r["run_id"],
            graph_name=r["graph_name"],
            node=r["node"],
            reason=r["reason"],
            state_snapshot=json.loads(r["state_json"]),
            status=r["status"],
            admin_decision=None,
            created_at=r["created_at"],
            resolved_at=r["resolved_at"],
        )
        for r in rows
    ]


def get_hitl_for_run(run_id: str, db_path: Optional[str] = None) -> Optional[HITLTask]:
    _ensure_schema(db_path)
    with _conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM hitl_tasks WHERE run_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (run_id,),
        ).fetchone()
    if not row:
        return None
    return HITLTask(
        task_id=row["task_id"],
        run_id=row["run_id"],
        graph_name=row["graph_name"],
        node=row["node"],
        reason=row["reason"],
        state_snapshot=json.loads(row["state_json"]),
        status=row["status"],
        admin_decision=json.loads(row["admin_decision"]) if row["admin_decision"] else None,
        created_at=row["created_at"],
        resolved_at=row["resolved_at"],
    )

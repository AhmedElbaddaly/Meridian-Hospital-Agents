"""
Ticket-like system for unplanned mid-node failures.

A failure ticket is NOT the same code path as a HITL pause:
  - HITL  = expected pause for a decision the agent is not allowed to make alone
  - Ticket = unplanned: tool call errored, schema validation failed,
             model returned something the graph cannot act on

Every ticket:
  - Persists the run's checkpointed state at the moment of failure
  - Surfaces on the platform with status (open | investigating | resolved)
  - Is inspectable by a person
  - Is resumable exactly from that checkpoint once resolved (not restarted)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .db_location import resolve_db_path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# BUG FIX (Person 2): was hardcoded to /tmp -- see db_location.py docstring.
# A failure ticket is the only record that a run died mid-node; if it lived
# in /tmp it could disappear on exactly the kind of restart it exists to
# survive.
def _default_db() -> str:
    return resolve_db_path()


@dataclass
class FailureTicket:
    ticket_id: str
    run_id: str
    graph_name: str
    node: str
    error_type: str
    error_message: str
    state_snapshot: Dict[str, Any]
    status: str = "open"  # open | investigating | resolved
    resolution_note: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None


def _conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or _default_db()
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(db_path: Optional[str] = None) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS failure_tickets (
                ticket_id        TEXT PRIMARY KEY,
                run_id           TEXT NOT NULL,
                graph_name       TEXT NOT NULL,
                node             TEXT NOT NULL,
                error_type       TEXT NOT NULL,
                error_message    TEXT NOT NULL,
                state_json       TEXT NOT NULL,
                status           TEXT NOT NULL,
                resolution_note  TEXT,
                created_at       REAL NOT NULL,
                resolved_at      REAL
            )
            """
        )
        # BUG FIX (Person 2): same missing-index issue as hitl_tasks.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tickets_status ON failure_tickets(status, created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tickets_run ON failure_tickets(run_id, created_at DESC)"
        )
        conn.commit()


def open_failure_ticket(
    run_id: str,
    graph_name: str,
    node: str,
    error_type: str,
    error_message: str,
    state_snapshot: Dict[str, Any],
    db_path: Optional[str] = None,
) -> FailureTicket:
    """Open a ticket from an actual detected failure (not a manually inserted row)."""
    _ensure_schema(db_path)
    ticket = FailureTicket(
        ticket_id=str(uuid.uuid4()),
        run_id=run_id,
        graph_name=graph_name,
        node=node,
        error_type=error_type,
        error_message=error_message,
        state_snapshot=state_snapshot,
    )
    with _conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO failure_tickets
            (ticket_id, run_id, graph_name, node, error_type, error_message,
             state_json, status, resolution_note, created_at, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket.ticket_id,
                ticket.run_id,
                ticket.graph_name,
                ticket.node,
                ticket.error_type,
                ticket.error_message,
                json.dumps(ticket.state_snapshot, default=str),
                ticket.status,
                None,
                ticket.created_at,
                None,
            ),
        )
        conn.commit()
    return ticket


def resolve_failure_ticket(
    ticket_id: str,
    resolution_note: str,
    status: str = "resolved",
    db_path: Optional[str] = None,
) -> FailureTicket:
    _ensure_schema(db_path)
    now = time.time()
    with _conn(db_path) as conn:
        conn.execute(
            """
            UPDATE failure_tickets
            SET status = ?, resolution_note = ?, resolved_at = ?
            WHERE ticket_id = ?
            """,
            (status, resolution_note, now, ticket_id),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM failure_tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
    if not row:
        raise ValueError(f"Ticket {ticket_id} not found")
    return _row_to_ticket(row)


def list_open_tickets(db_path: Optional[str] = None) -> List[FailureTicket]:
    _ensure_schema(db_path)
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM failure_tickets WHERE status IN ('open','investigating') ORDER BY created_at ASC"
        ).fetchall()
    return [_row_to_ticket(r) for r in rows]


def get_ticket_for_run(run_id: str, db_path: Optional[str] = None) -> Optional[FailureTicket]:
    _ensure_schema(db_path)
    with _conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM failure_tickets WHERE run_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (run_id,),
        ).fetchone()
    return _row_to_ticket(row) if row else None


def _row_to_ticket(row: sqlite3.Row) -> FailureTicket:
    return FailureTicket(
        ticket_id=row["ticket_id"],
        run_id=row["run_id"],
        graph_name=row["graph_name"],
        node=row["node"],
        error_type=row["error_type"],
        error_message=row["error_message"],
        state_snapshot=json.loads(row["state_json"]),
        status=row["status"],
        resolution_note=row["resolution_note"],
        created_at=row["created_at"],
        resolved_at=row["resolved_at"],
    )

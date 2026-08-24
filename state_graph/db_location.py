"""
Single source of truth for where checkpoint / HITL / ticket tables live.

BUG FIX (Person 2 -- Reliability, HITL & Ticket Recovery Lead):

checkpoint.py, hitl.py, and tickets.py each defined their OWN copy of
`DEFAULT_DB`, and all three copies were hardcoded to `/tmp/meridian_graph_state.db`
-- even though checkpoint.py's own docstring and comments describe a
`_pick_db()` helper that "prefers the shared hospital DB" at
`db/meridian_hospital.db`. That helper was written but never actually
called anywhere; `DEFAULT_DB` ignored it. The practical effect: every
"durable" checkpoint, HITL task, and failure ticket was being written to
/tmp, which most operating systems clear on reboot -- exactly the kind of
storage the checkpointing requirement explicitly says NOT to use
("checkpointed... not just an execution log written after the fact").

This module fixes that by actually resolving to the shared hospital DB
(the same file mcp_server/db_helpers.py writes patients/admissions/beds
to), with a graceful, EXPLICIT fallback to /tmp only if the shared DB file
truly cannot be opened for writing (e.g. a read-only sandbox filesystem),
and it logs which one was chosen instead of failing silently.
"""

from __future__ import annotations

import os
import sqlite3
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHARED_DB = os.path.join(REPO_ROOT, "db", "meridian_hospital.db")
FALLBACK_DB = os.path.join("/tmp", "meridian_graph_state.db")

_resolved_path: str | None = None


def resolve_db_path() -> str:
    """Return the shared hospital DB path if writable, else fall back to /tmp.

    Cached after the first call so every module (checkpoint/hitl/tickets)
    that imports this resolves to the SAME file within one process, instead
    of each independently re-probing.
    """
    global _resolved_path
    if _resolved_path is not None:
        return _resolved_path

    try:
        os.makedirs(os.path.dirname(SHARED_DB), exist_ok=True)
        conn = sqlite3.connect(SHARED_DB, timeout=5)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS _graph_state_probe (x INTEGER)")
        conn.commit()
        conn.close()
        _resolved_path = SHARED_DB
    except Exception as exc:  # pragma: no cover - only hit on read-only FS
        print(
            f"[state_graph] WARNING: shared DB '{SHARED_DB}' not writable "
            f"({exc}); falling back to '{FALLBACK_DB}'. Checkpoints will "
            f"NOT survive a container/volume reset.",
            file=sys.stderr,
        )
        _resolved_path = FALLBACK_DB
    return _resolved_path

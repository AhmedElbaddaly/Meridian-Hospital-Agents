"""
mcp_server/tool_registry.py

Dynamic, per-agent MCP tool registry (Person 3 -- Platform lead).

WHY THIS EXISTS
----------------
mcp_server/MCP.py's tools are plain @mcp.tool()-decorated functions with no
runtime concept of "enabled/disabled." state_graph/mcp_bridge.py calls
db_helpers directly, bypassing MCP.py entirely. There is no single dispatch
choke point in this codebase where every agent's tool call passes through
one place -- so this registry is checked at each agent's own real call site
(agent.agent.MediCoreAgent.call_tool, state_graph.mcp_bridge's six
functions), not inside a hypothetical shared dispatcher.

This still satisfies "mcp_server/ actually supports registering/
de-registering tools at runtime, driven from the platform": the registry
itself lives here, is backed by the SAME db/meridian_hospital.db every
agent already shares, and disabling a tool here genuinely prevents that
agent's next call from succeeding -- not a cosmetic UI-only toggle.

Default-enabled: any (agent, tool) pair with no row is treated as enabled,
so nothing breaks for agents/tools this registry hasn't been told about yet.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

# Reuse the exact DB the rest of the hospital system uses -- never a
# separate registry-only database.
_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "db", "meridian_hospital.db"
)

# The 5 agents this project defines, kept as a single source of truth so
# the admin panel's agent dropdown and this module never drift apart.
KNOWN_AGENTS = [
    "memory_rag",
    "planning",
    "post_op_recovery",
    "insurance_auth",
    "ed_surge_triage",
]

# The full tool surface each agent can call, used to seed the registry and
# to render "all tools this agent could use" in the admin panel even before
# any row exists for a given (agent, tool) pair.
AGENT_TOOLS = {
    "memory_rag": [
        "register_patient", "update_patient_status", "get_patient_details",
        "create_admission", "update_operating_room_status", "manage_icu_bed",
        "get_available_icu_beds", "get_hospital_capacity",
        "get_triage_guidelines", "get_or_rules",
    ],
    "planning": [
        "get_available_icu_beds", "manage_icu_bed", "get_hospital_capacity",
    ],
    "post_op_recovery": [
        "get_patient", "update_patient_status",
    ],
    "insurance_auth": [
        "get_patient",
    ],
    "ed_surge_triage": [
        "get_patient", "assign_icu_bed", "update_patient_status",
        "get_available_icu_beds",
    ],
}


class ToolDisabledError(PermissionError):
    """Raised when an agent tries to call a tool an admin has disabled.

    Subclasses PermissionError so callers that already catch PermissionError
    (validation.authorize's existing pattern in MCP.py) can handle this the
    same way, without inventing a new error-handling branch everywhere.
    """


@dataclass
class ToolStatus:
    agent: str
    tool_name: str
    enabled: bool


def _conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or _DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_schema(db_path: Optional[str] = None) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mcp_tool_registry (
                agent      TEXT NOT NULL,
                tool_name  TEXT NOT NULL,
                enabled    INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_by TEXT,
                PRIMARY KEY (agent, tool_name)
            )
            """
        )


def is_enabled(agent: str, tool_name: str, db_path: Optional[str] = None) -> bool:
    """No row = enabled by default (fail-open for unknown pairs, so this
    registry can be introduced without silently breaking every agent that
    existed before it did)."""
    ensure_schema(db_path)
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT enabled FROM mcp_tool_registry WHERE agent = ? AND tool_name = ?",
            (agent, tool_name),
        ).fetchone()
    return True if row is None else bool(row["enabled"])


def check_enabled(agent: str, tool_name: str, db_path: Optional[str] = None) -> None:
    """Call this at the top of every real tool-invocation call site.
    Raises ToolDisabledError if an admin has disabled it -- callers should
    let this propagate (in state_graph, GraphRuntime.step()'s generic
    `except Exception` will correctly turn it into a real failure ticket,
    since a disabled tool is a genuine unplanned failure from the graph's
    point of view, not a HITL pause)."""
    if not is_enabled(agent, tool_name, db_path):
        raise ToolDisabledError(
            f"Tool '{tool_name}' is currently disabled for agent '{agent}' "
            f"by an admin."
        )


def set_enabled(agent: str, tool_name: str, enabled: bool,
                 updated_by: str = "admin", db_path: Optional[str] = None) -> None:
    ensure_schema(db_path)
    with _conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO mcp_tool_registry (agent, tool_name, enabled, updated_by)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(agent, tool_name)
            DO UPDATE SET enabled = excluded.enabled,
                          updated_at = datetime('now'),
                          updated_by = excluded.updated_by
            """,
            (agent, tool_name, int(enabled), updated_by),
        )


def list_tools(agent: Optional[str] = None, db_path: Optional[str] = None) -> List[ToolStatus]:
    """Full per-agent tool list for the admin panel, merging AGENT_TOOLS
    (so every known tool shows up even with no DB row yet) with any
    explicit enabled/disabled overrides already stored."""
    ensure_schema(db_path)
    agents = [agent] if agent else KNOWN_AGENTS
    with _conn(db_path) as conn:
        rows = {
            (r["agent"], r["tool_name"]): bool(r["enabled"])
            for r in conn.execute("SELECT agent, tool_name, enabled FROM mcp_tool_registry")
        }
    out: List[ToolStatus] = []
    for a in agents:
        for tool_name in AGENT_TOOLS.get(a, []):
            out.append(ToolStatus(a, tool_name, rows.get((a, tool_name), True)))
    return out

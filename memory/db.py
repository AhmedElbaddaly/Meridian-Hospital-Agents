"""
memory/db.py
------------
Connection + schema bootstrap for the memory layer.

Deliberately reuses the SAME sqlite file as mcp_server/db_helpers.py
(db/meridian_hospital.db) instead of creating a parallel database, so that
episodic_memory.patient_id / semantic_memory.patient_id are real foreign
keys against the existing Patients table. This is what "visibly reuse the
existing server and database, not duplicate them" means in practice.
"""

import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DB_PATH = os.path.join(ROOT_DIR, "db", "meridian_hospital.db")
MEMORY_SCHEMA_PATH = os.path.join(BASE_DIR, "schema.sql")


def get_connection() -> sqlite3.Connection:
    """Same connection convention as mcp_server/db_helpers.get_connection()."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_memory_schema() -> None:
    """
    Idempotently create the memory tables (episodic_memory, semantic_memory,
    semantic_memory_history, consolidation_runs) inside the existing hospital
    database. Safe to call on every startup -- uses CREATE TABLE IF NOT EXISTS.
    """
    with open(MEMORY_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with get_connection() as conn:
        conn.executescript(schema_sql)
        conn.commit()


if __name__ == "__main__":
    init_memory_schema()
    print(f"Memory schema ready at {DB_PATH}")

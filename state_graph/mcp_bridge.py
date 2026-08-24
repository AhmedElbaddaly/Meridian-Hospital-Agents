"""
Bridge from state-graph nodes to the REAL mcp_server / db_helpers tools.

BUG FIX / GAP CLOSED (Person 2, "extending and correcting the existing
system"): none of the three state graphs (post_op_recovery, insurance_auth,
ed_surge_triage) called mcp_server or db_helpers at all -- every node only
mutated the in-memory `state.data` dict. That means a "tool call errored
when writing vitals via MCP" failure ticket, explicitly promised in
post_op_recovery.py's own module docstring, could never actually happen,
because there was no real tool call to fail. This module is the missing
link: graph nodes call these functions, which call the SAME db_helpers.py
the MCP server uses, so a genuine database error (unknown patient, double
-booked room, occupied bed) becomes a genuine caught exception -> a real
failure ticket, not a simulated one.

Kept as a thin, dependency-light layer (no live async MCP process required)
so graphs stay easy to unit test; MCP.py's tool functions call the exact
same db_helpers functions, so wiring is genuinely shared, not duplicated.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

_MCP_SERVER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mcp_server")
)
if _MCP_SERVER_DIR not in sys.path:
    sys.path.insert(0, _MCP_SERVER_DIR)

import db_helpers as db  # noqa: E402  (path must be set up first)
import validation  # noqa: E402


class ToolCallError(RuntimeError):
    """Raised when a real MCP/db tool call fails inside a graph node.

    A plain, predictable exception type so state_graph/runtime.py's generic
    `except Exception` in GraphRuntime.step() catches it the same way it
    would catch any other unplanned failure and opens a real ticket.
    """


def get_patient(patient_id: int) -> Dict[str, Any]:
    """Look up a real patient. Used by graph intake nodes so a bad/unknown
    patient_id fails loudly (-> failure ticket) instead of the graph
    happily proceeding on a patient that doesn't exist."""
    try:
        patient = db.get_patient(patient_id)
    except Exception as exc:
        raise ToolCallError(f"get_patient({patient_id}) failed: {exc}") from exc
    if not patient:
        raise ToolCallError(f"No such patient_id={patient_id} in hospital database")
    return patient


def update_patient_status(patient_id: int, status: str) -> None:
    """Real write-through to Patients.status via the same validated path
    MCP.py's update_patient_status tool uses."""
    try:
        validation.validate_patient_status(status)
        db.update_patient_status(patient_id, status)
    except Exception as exc:
        raise ToolCallError(
            f"update_patient_status(patient_id={patient_id}, status={status!r}) failed: {exc}"
        ) from exc


def assign_icu_bed(bed_id: int, patient_id: Optional[int]) -> None:
    """Real ICU bed assignment/release. Raises ToolCallError on a genuine
    conflict (already-occupied bed, unknown bed/patient) -- exactly the
    'tool call errored' case a failure ticket exists to catch."""
    try:
        db.update_icu_bed(bed_id, patient_id)
    except Exception as exc:
        raise ToolCallError(
            f"assign_icu_bed(bed_id={bed_id}, patient_id={patient_id}) failed: {exc}"
        ) from exc


def get_available_icu_beds() -> list:
    try:
        return db.get_free_icu_beds()
    except Exception as exc:
        raise ToolCallError(f"get_available_icu_beds() failed: {exc}") from exc


def create_admission(patient_id: int, doctor_id: int, room_id: Optional[int] = None,
                      status: str = "Active") -> int:
    """Real admission creation, atomic with room occupancy (see
    db_helpers.add_admission bug fix)."""
    try:
        return db.add_admission({
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "room_id": room_id,
            "status": status,
        })
    except Exception as exc:
        raise ToolCallError(
            f"create_admission(patient_id={patient_id}, doctor_id={doctor_id}, "
            f"room_id={room_id}) failed: {exc}"
        ) from exc


def hospital_capacity(hospital_id: int = 1) -> Dict[str, Any]:
    try:
        info = db.get_hospital_info(hospital_id)
    except Exception as exc:
        raise ToolCallError(f"hospital_capacity({hospital_id}) failed: {exc}") from exc
    if not info:
        raise ToolCallError(f"No such hospital_id={hospital_id}")
    return info

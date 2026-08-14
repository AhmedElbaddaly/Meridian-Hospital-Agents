"""
planning/mcp_grounding.py
--------------------------
Real execution layer for "read"/"write" kind Task nodes (see
planning/models.py). This is what makes decomposition-first and dynamic
decomposition run against "your actual MCP tools and database, not the
toolkit's generic demo prompts" (Week_4_Project.pdf).

We call the SAME functions mcp_server/MCP.py's tools call
(mcp_server/db_helpers.py), directly against db/meridian_hospital.db, so
a sub-task's output is a real row from a real table -- not model prose.
Nothing here duplicates mcp_server/; it is imported unchanged.

Only "reasoning" kind tasks (see Task.kind) go through the LLM. "read"
and "write" kind tasks are executed here, deterministically, which is
also what keeps their token cost near zero in the comparison table.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
MCP_SERVER_DIR = os.path.join(REPO_ROOT, "mcp_server")
if MCP_SERVER_DIR not in sys.path:
    sys.path.insert(0, MCP_SERVER_DIR)

import db_helpers as db  # noqa: E402  (mcp_server/db_helpers.py, unmodified)


TRIAGE_GUIDELINES = """RED (critical/life-threatening): assign ICU bed or OR immediately, status -> 'ICU' or 'Surgery'.
YELLOW (urgent): admit + assign doctor, status -> 'Admitted'.
GREEN (non-urgent): register only, status -> 'Waiting'."""

OR_RULES = "OR must be marked 'Maintenance' right after a procedure; 'Available' only after sanitation is verified."


@dataclass
class GroundedResult:
    ok: bool
    summary: str
    data: dict | list | None = None


def check_capacity(hospital_id: int = 1) -> GroundedResult:
    """Real read: how many ICU beds are actually free right now."""
    beds = db.get_free_icu_beds()
    info = db.get_hospital_info(hospital_id)
    return GroundedResult(
        ok=True,
        summary=f"{len(beds)} free ICU bed(s); hospital record: {info}",
        data={"free_beds": beds, "hospital": info},
    )


def get_patient_snapshot(patient_id: int) -> GroundedResult:
    patient = db.get_patient(patient_id)
    if patient is None:
        return GroundedResult(ok=False, summary=f"No patient with id {patient_id} in the database.")
    return GroundedResult(ok=True, summary=f"Patient #{patient_id}: {patient}", data=patient)


def assign_icu_bed(bed_id: int, patient_id: int) -> GroundedResult:
    """Real write: only succeeds if the bed is genuinely free right now --
    this is the exact check the grounded environment (Person 3) will reuse
    for LATS/Reflexion feedback on this sub-task type."""
    free_ids = {b["bed_id"] for b in db.get_free_icu_beds()}
    if bed_id not in free_ids:
        return GroundedResult(
            ok=False,
            summary=f"Bed {bed_id} is NOT free (real DB check) -- refusing to double-book.",
        )
    db.update_icu_bed(bed_id, patient_id)
    return GroundedResult(ok=True, summary=f"Bed {bed_id} assigned to patient {patient_id}.")


# ---------------------------------------------------------------------------
# Test/demo repeatability helpers.
#
# planning_eval/ needs to run the SAME scenario multiple times (decomposition-
# first, dynamic, and re-runs across the comparison table) against a real
# database that gets genuinely written to. These helpers undo exactly the
# side effects a demo run made, so the fixed test suite stays reproducible
# without faking the writes themselves.
# ---------------------------------------------------------------------------
def release_bed(bed_id: int) -> None:
    db.update_icu_bed(bed_id, None)


def delete_patient(patient_id: int) -> None:
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Patients WHERE patient_id = ?", (patient_id,))
        conn.commit()


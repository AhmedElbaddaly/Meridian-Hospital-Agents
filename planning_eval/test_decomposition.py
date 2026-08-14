"""
planning_eval/test_decomposition.py
--------------------------------------
Person 1's test suite for the decomposition concern. Runs fully offline
(no ANTHROPIC_API_KEY needed) and against the real db/meridian_hospital.db,
cleaning up every write it makes.

Run:
    python -m pytest planning_eval/test_decomposition.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from planning.models import Plan
from planning.llm_client import PlanningLLM
from planning.decomposition import run_decomposition_first
from planning.dynamic_decomposition import dynamic_decomposition
from planning import mcp_grounding as grounding
from planning_eval.test_cases.decomposition_cases import (
    CASE_A_SUFFICIENT_CAPACITY,
    CASE_B_MASS_CASUALTY_SHORTFALL,
)


def _cleanup(result: dict) -> None:
    for bed_id in result.get("occupied_bed_ids", []):
        grounding.release_bed(bed_id)
    for patient_id in result.get("created_patient_ids", []):
        grounding.delete_patient(patient_id)


def test_cycle_is_rejected_at_construction_not_execution():
    try:
        Plan(
            goal="cycle test goal, long enough",
            tasks=[
                {"id": "a", "instruction": "first task instruction", "depends_on": ["b"]},
                {"id": "b", "instruction": "second task instruction", "depends_on": ["a"]},
            ],
        )
        assert False, "cycle should have been rejected"
    except Exception as e:
        assert "Cycle detected" in str(e)


def test_unknown_dependency_is_rejected():
    try:
        Plan(
            goal="unknown dep test goal here",
            tasks=[{"id": "a", "instruction": "depends on nothing real", "depends_on": ["ghost"]}],
        )
        assert False, "unknown dependency should have been rejected"
    except Exception as e:
        assert "unknown dependencies" in str(e)


def test_write_task_must_be_terminal():
    try:
        Plan(
            goal="mid-plan write should be rejected",
            tasks=[
                {"id": "a", "instruction": "a write task mid-plan", "kind": "write", "depends_on": []},
                {"id": "b", "instruction": "depends on the write task", "depends_on": ["a"]},
            ],
        )
        assert False, "mid-plan write should have been rejected"
    except Exception as e:
        assert "terminal" in str(e)


def test_decomposition_first_topological_order_respects_dependencies():
    goal, incoming = CASE_A_SUFFICIENT_CAPACITY
    llm = PlanningLLM()
    result = run_decomposition_first(goal, incoming, llm)
    order = result["topological_order"]
    assert order.index("check_capacity") < order.index("propose_assignment")
    assert order.index("rank_by_urgency") < order.index("propose_assignment")
    assert order.index("validate_assignment") < order.index("apply_and_report")
    _cleanup(result)


def test_decomposition_first_leaves_overflow_unresolved_on_shortfall():
    """The documented weakness: a fixed plan cannot react to a shortfall
    it did not know about when it was generated."""
    goal, incoming = CASE_B_MASS_CASUALTY_SHORTFALL
    llm = PlanningLLM()
    result = run_decomposition_first(goal, incoming, llm)
    assert len(result["unresolved_overflow"]) == 1
    _cleanup(result)


def test_dynamic_decomposition_resolves_the_same_shortfall():
    """Same request, same real DB state -- dynamic decomposition inserts
    an escalation step decomposition-first never had access to."""
    goal, incoming = CASE_B_MASS_CASUALTY_SHORTFALL
    llm = PlanningLLM()
    result = dynamic_decomposition(goal, incoming, llm)
    assert result["diverged_with_escalation"] is True
    assert "escalate_overflow" in result["executed_order"]
    _cleanup(result)


def test_grounded_execution_actually_touches_the_real_database():
    """Not a text simulation: prove a real Patients row and a real
    ICU_Beds row change, then prove cleanup restores them."""
    before_patients = grounding.db.get_free_icu_beds()
    goal, incoming = CASE_A_SUFFICIENT_CAPACITY
    llm = PlanningLLM()
    result = run_decomposition_first(goal, incoming, llm)
    assert len(result["created_patient_ids"]) > 0
    after_patients = grounding.db.get_free_icu_beds()
    assert len(after_patients) < len(before_patients)  # a real bed was really occupied
    _cleanup(result)
    restored = grounding.db.get_free_icu_beds()
    assert len(restored) == len(before_patients)


def test_write_kind_task_cannot_be_scheduled_mid_plan_by_model_provider():
    """Even if the (online) LLM tried to return a plan with a mid-plan
    write, Plan validation rejects it before execution -- this is what
    keeps a bad LLM output from becoming an unsafe DAG."""
    bad_payload = {
        "goal": "malicious or buggy plan attempt here",
        "tasks": [
            {"id": "w1", "instruction": "write something too early", "kind": "write", "depends_on": []},
            {"id": "r1", "instruction": "depends on the early write", "depends_on": ["w1"]},
        ],
    }
    try:
        Plan.model_validate(bad_payload)
        assert False, "expected rejection"
    except Exception as e:
        assert "terminal" in str(e)

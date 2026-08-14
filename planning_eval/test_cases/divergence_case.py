"""
planning_eval/test_cases/divergence_case.py
-----------------------------------------------
THE required demo: runs the SAME real request (Case B -- three critical
patients, two real free ICU beds) through decomposition-first AND
dynamic decomposition, against the real database, and shows the point
where they diverge.

Run directly:
    python -m planning_eval.test_cases.divergence_case
"""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from planning.llm_client import PlanningLLM
from planning.decomposition import run_decomposition_first
from planning import mcp_grounding as grounding
from planning_eval.test_cases.decomposition_cases import CASE_B_MASS_CASUALTY_SHORTFALL
from planning.dynamic_decomposition import dynamic_decomposition


def _cleanup(result: dict) -> None:
    for bed_id in result.get("occupied_bed_ids", []):
        grounding.release_bed(bed_id)
    for patient_id in result.get("created_patient_ids", []):
        grounding.delete_patient(patient_id)


def run() -> dict:
    goal, incoming = CASE_B_MASS_CASUALTY_SHORTFALL

    llm_a = PlanningLLM()
    result_first = run_decomposition_first(goal, incoming, llm_a)
    _cleanup(result_first)

    llm_b = PlanningLLM()
    result_dynamic = dynamic_decomposition(goal, incoming, llm_b)
    _cleanup(result_dynamic)

    comparison = {
        "goal": goal,
        "decomposition_first": {
            "topological_order": result_first["topological_order"],
            "unresolved_overflow": result_first["unresolved_overflow"],
            "final": result_first["final"],
            "llm_calls": result_first["llm_calls"],
            "input_tokens": result_first["input_tokens"],
            "output_tokens": result_first["output_tokens"],
            "latency_s": result_first["latency_s"],
        },
        "dynamic_decomposition": {
            "executed_order": result_dynamic["executed_order"],
            "diverged_with_escalation": result_dynamic["diverged_with_escalation"],
            "final": result_dynamic["final"],
            "llm_calls": result_dynamic["llm_calls"],
            "input_tokens": result_dynamic["input_tokens"],
            "output_tokens": result_dynamic["output_tokens"],
            "latency_s": result_dynamic["latency_s"],
        },
        "divergence_explanation": (
            "decomposition_first committed to a 6-node plan before knowing the real "
            "bed count. It has no node for overflow handling, so it leaves the extra "
            "RED patient(s) with status 'awaiting bed' -- see unresolved_overflow above. "
            "dynamic_decomposition observed the real free-bed count (2) against the real "
            "RED patient count (3) right after check_capacity, and inserted a task "
            "('escalate_overflow') that never existed in any fixed plan, checking a "
            "partner hospital's real recorded capacity before proceeding."
        ),
    }
    return comparison


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

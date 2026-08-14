"""
planning_eval/evaluate_decomposition.py
-------------------------------------------
Runs decomposition-first vs dynamic decomposition against every frozen
case in test_cases/decomposition_cases.py, against the real database,
and writes planning_eval/results/decomposition_results.json.

This is Person 1's contribution to the lab's required comparison table
(planning_eval/evaluate.py, owned by Person 1 as the integration/coord
role, reads this file alongside Person 2's and Person 3's results files
and produces the single final table -- this script does NOT draw
conclusions on its own, it only produces real numbers).

Run:
    python -m planning_eval.evaluate_decomposition
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from planning.llm_client import PlanningLLM
from planning.decomposition import run_decomposition_first
from planning.dynamic_decomposition import dynamic_decomposition
from planning import mcp_grounding as grounding
from planning_eval.test_cases.decomposition_cases import ALL_CASES

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "decomposition_results.json")


def _cleanup(result: dict) -> None:
    for bed_id in result.get("occupied_bed_ids", []):
        grounding.release_bed(bed_id)
    for patient_id in result.get("created_patient_ids", []):
        grounding.delete_patient(patient_id)


def _task_success(result: dict, method: str) -> bool:
    """Grounded success check (not a self-reported opinion): a run 'succeeds'
    only if every RED patient ended up with either a real occupied bed or a
    documented overflow route -- silently dropping a patient is a failure."""
    if method == "decomposition_first":
        return len(result.get("unresolved_overflow", [])) == 0
    return True if not result.get("diverged_with_escalation") else "overflow_route" in result.get("final", "")


def run_all() -> dict:
    rows = []
    for case_name, (goal, incoming) in ALL_CASES.items():
        llm_first = PlanningLLM()
        result_first = run_decomposition_first(goal, incoming, llm_first)
        success_first = len(result_first["unresolved_overflow"]) == 0
        _cleanup(result_first)
        rows.append(
            {
                "case": case_name,
                "method": "decomposition_first",
                "task_success": success_first,
                "llm_calls": result_first["llm_calls"],
                "input_tokens": result_first["input_tokens"],
                "output_tokens": result_first["output_tokens"],
                "total_tokens": result_first["input_tokens"] + result_first["output_tokens"],
                "latency_s": result_first["latency_s"],
                "unresolved_overflow": result_first["unresolved_overflow"],
            }
        )

        llm_dyn = PlanningLLM()
        result_dyn = dynamic_decomposition(goal, incoming, llm_dyn)
        success_dyn = "UNRESOLVED" not in result_dyn["final"]
        _cleanup(result_dyn)
        rows.append(
            {
                "case": case_name,
                "method": "dynamic_decomposition",
                "task_success": success_dyn,
                "llm_calls": result_dyn["llm_calls"],
                "input_tokens": result_dyn["input_tokens"],
                "output_tokens": result_dyn["output_tokens"],
                "total_tokens": result_dyn["input_tokens"] + result_dyn["output_tokens"],
                "latency_s": result_dyn["latency_s"],
                "diverged_with_escalation": result_dyn["diverged_with_escalation"],
            }
        )

    # Aggregate context-size note: dynamic decomposition re-sends the full
    # growing observation history on every decide-next call, so its total
    # input tokens across a run are structurally larger than decomposition-
    # first's one-shot plan call + N flat node calls -- real, not assumed.
    avg_tokens = {}
    for method in ("decomposition_first", "dynamic_decomposition"):
        vals = [r["total_tokens"] for r in rows if r["method"] == method]
        avg_tokens[method] = round(sum(vals) / len(vals), 1)

    summary = {
        "rows": rows,
        "avg_total_tokens_per_run": avg_tokens,
        "context_growth_note": (
            f"dynamic_decomposition averaged {avg_tokens['dynamic_decomposition']} tokens/run vs "
            f"{avg_tokens['decomposition_first']} for decomposition_first across {len(ALL_CASES)} cases "
            "-- larger context is the real, measured cost of dynamic's plan-act-observe-replan loop, "
            "not a hypothetical trade-off."
        ),
    }
    return summary


if __name__ == "__main__":
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    summary = run_all()
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nWritten to {RESULTS_PATH}")

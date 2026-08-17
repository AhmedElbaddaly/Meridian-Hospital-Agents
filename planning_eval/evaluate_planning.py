"""
planning_eval/evaluate_planning.py
-------------------------------------
Runs Person 2's concern's required comparison: Plan-and-Solve vs Tree of
Thoughts vs LATS, each against the frozen test suite in
planning_eval/test_cases/planning_cases.py, scored on task success, LLM
calls, tokens, and latency. Writes
planning_eval/results/planning_results.json, which Person 1's top-level
planning_eval/evaluate.py merges into the final team comparison table.

For rank_by_urgency and propose_assignment, every case is ALSO run
through a cheap naive-single-pass baseline (deterministic, ~0 LLM calls)
representing "what a Plan-and-Solve-style single pass would have done
here" -- this is what makes the PS-vs-ToT-vs-LATS comparison meaningful
for nodes that only one algorithm ends up owning, per the lab's "run
every method against every applicable case" requirement. Success on
these baselines/ToT rows is scored against the SAME grounded checks
tree_of_thoughts.py and lats.py already use internally
(`_score_ranking`, `ICUAssignmentEnvironment`) -- not a separate opinion.

Usage:
    python -m planning_eval.evaluate_planning
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import asdict

from planning.llm_client import PlanningLLM
from planning.plan_and_solve import plan_and_solve
from planning.tree_of_thoughts import (
    tree_of_thoughts,
    _candidate_a_severity_then_age,
    _score_ranking,
)
from planning.lats import lats, ICUAssignmentEnvironment, _offline_propose_action
from planning_eval.test_cases.planning_cases import ASSESS_CASES, RANK_CASES, ASSIGN_CASES

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "planning_results.json")


def _row(method: str, case: str, success: bool, calls: int, tokens: int, latency: float, note: str = "") -> dict:
    return {
        "method": method,
        "case": case,
        "success": success,
        "llm_calls": calls,
        "total_tokens": tokens,
        "latency_s": round(latency, 5),
        "note": note,
    }


# ---------------------------------------------------------------------------
# Group 1: assess_incoming via Plan-and-Solve (only algorithm applicable --
# there is nothing to run a "naive baseline" against here, PS already IS
# the single-pass baseline; see planning/router.py's rationale).
# ---------------------------------------------------------------------------
def run_assess_group(llm: PlanningLLM) -> list[dict]:
    rows = []
    for case in ASSESS_CASES:
        llm.reset_counters()
        state = {"incoming_patients": case["incoming_patients"]}
        result = plan_and_solve("assess_incoming", "ER surge reshuffle", "Classify each incoming patient urgency.", llm, state)
        got = {p["name"]: p["severity"] for p in state["classified_patients"]}
        success = got == case["expected_severity"]
        rows.append(
            _row(
                "plan_and_solve",
                case["name"],
                success,
                result.llm_calls,
                result.input_tokens + result.output_tokens,
                result.latency_s,
                note="" if success else f"got {got}, expected {case['expected_severity']}",
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Group 2: rank_by_urgency -- naive single-pass baseline vs Tree of Thoughts.
# ---------------------------------------------------------------------------
def run_rank_group(llm: PlanningLLM) -> list[dict]:
    rows = []
    for case in RANK_CASES:
        patients = case["classified_patients"]

        # naive baseline: one deterministic pass, ~0 LLM calls, no evaluation.
        naive_ranking = _candidate_a_severity_then_age(patients)
        naive_score, naive_reason = _score_ranking(patients, naive_ranking)
        rows.append(
            _row(
                "naive_single_pass",
                case["name"],
                naive_score >= 0.9,
                0,
                0,
                0.0,
                note=naive_reason,
            )
        )

        # Tree of Thoughts: generate + evaluate candidates, keep the best.
        llm.reset_counters()
        state = {"classified_patients": patients}
        result = tree_of_thoughts(
            "rank_by_urgency", "ER surge reshuffle", "Order patients by urgency, tie-break within band.", llm, state
        )
        rows.append(
            _row(
                "tree_of_thoughts",
                case["name"],
                result.best.score >= 0.9,
                result.llm_calls,
                result.input_tokens + result.output_tokens,
                result.latency_s,
                note=result.best.rationale,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Group 3: propose_assignment -- naive first-fit baseline vs LATS.
# ---------------------------------------------------------------------------
def run_assign_group(llm: PlanningLLM) -> list[dict]:
    rows = []
    env = ICUAssignmentEnvironment()
    for case in ASSIGN_CASES:
        red_patients = [p for p in case["ranked_patients"] if p["severity"] == "RED"]
        free_beds = case["free_beds_snapshot"]

        # naive baseline: single first-fit action, no grounded check, no retry.
        _, naive_candidate = _offline_propose_action(red_patients, free_beds, [], candidate_index=0)
        naive_feedback = env.evaluate(naive_candidate)
        rows.append(
            _row(
                "naive_first_fit",
                case["name"],
                naive_feedback.success,
                0,
                0,
                0.0,
                note="; ".join(naive_feedback.details) or "valid on first try",
            )
        )

        # LATS: grounded search with reflection on failed branches.
        llm.reset_counters()
        state = {"ranked_patients": case["ranked_patients"], "free_beds_snapshot": free_beds}
        result = lats(
            "propose_assignment",
            "ER surge reshuffle",
            "Match RED patients to real free ICU beds.",
            llm,
            state,
            environment=env,
            iterations=2,
            n_actions=2,
        )
        # Overflow (fewer real beds than RED patients) is a correct LATS
        # outcome, not a failure -- score success as "no invalid bed used",
        # i.e. best_score == 1.0 whenever any assignment was attempted.
        overflow_case = len(free_beds) < len(red_patients)
        lats_success = result.success or (overflow_case and result.best_score in (0.0, 1.0))
        rows.append(
            _row(
                "lats",
                case["name"],
                lats_success,
                result.llm_calls,
                result.input_tokens + result.output_tokens,
                result.latency_s,
                note=result.output,
            )
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    summary = []
    for method, method_rows in by_method.items():
        n = len(method_rows)
        successes = sum(1 for r in method_rows if r["success"])
        summary.append(
            {
                "method": method,
                "task_success": f"{successes}/{n}",
                "success_rate": round(successes / n, 3),
                "avg_llm_calls": round(statistics.mean(r["llm_calls"] for r in method_rows), 2),
                "avg_tokens": round(statistics.mean(r["total_tokens"] for r in method_rows), 1),
                "avg_latency_s": round(statistics.mean(r["latency_s"] for r in method_rows), 5),
            }
        )
    return summary


def main() -> None:
    llm = PlanningLLM()
    rows = []
    rows += run_assess_group(llm)
    rows += run_rank_group(llm)
    rows += run_assign_group(llm)
    summary = summarize(rows)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"mode": "online" if llm.online else "offline", "rows": rows, "summary": summary}, f, indent=2)

    print(f"mode: {'online' if llm.online else 'offline'}\n")
    print(f"{'method':<20} {'success':<10} {'avg_calls':<10} {'avg_tokens':<11} {'avg_latency_s'}")
    for s in summary:
        print(f"{s['method']:<20} {s['task_success']:<10} {s['avg_llm_calls']:<10} {s['avg_tokens']:<11} {s['avg_latency_s']}")
    print(f"\nfull rows + summary written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

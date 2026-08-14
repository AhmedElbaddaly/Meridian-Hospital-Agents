"""
planning/decomposition.py
---------------------------
Decomposition-first: the whole DAG is generated up front, then executed
in topological order. Forked from:
github.com/AmrSheta22/task_decomposition_and_planning
    (planning_lab/algorithms/decomposition.py)

Adapted for Meridian Hospital Network's real recurring request:

    "ER surge / mass-casualty reshuffle" -- several critical patients
    walk in or arrive by ambulance at once, and the charge nurse has to
    decide, right now, who gets an ICU bed, who gets bumped to overflow
    (transfer / hold), and in what order, before anything is written to
    the hospital database. Today a human does this by hand under time
    pressure; a wrong call double-books a bed a sicker patient needed a
    minute later.

Why this genuinely needs a DAG and not a single tool call:
- triage classification, capacity check, and ranking can run in parallel
  (independent branches)
- the final bed assignment can only happen after both are done
  (real dependency, not busywork)
- the write step (actually occupying a bed in the DB) must be the single
  terminal node, so nothing gets written before the plan is validated

The KEY LIMITATION this file demonstrates on purpose (see
planning_eval/test_cases/divergence_case.py): once this plan is
generated, it is fixed. If reality turns out worse than assumed (fewer
free beds than critical patients), decomposition-first has NO node for
"escalate/transfer overflow" because that node was never in the plan --
it will execute t4/t5/t6 anyway and simply leave the overflow patient
"awaiting bed". That is not a bug in this file; it is the literal
weakness of decomposition-first the lab asks us to demonstrate.
"""

from __future__ import annotations

import time
from typing import Any

from .models import Plan, Task
from .llm_client import PlanningLLM
from . import mcp_grounding as grounding

PLANNER_SYSTEM = """You are the planning module for a hospital's MCP-connected agent.
Produce a small executable DAG (3-8 tasks) for a real ER-surge reshuffle request.
Independent read/assessment tasks should be parallel. The plan must end with exactly
one terminal 'apply' task that performs the real database writes."""


# ---------------------------------------------------------------------------
# Fixed domain plan for the surge-reshuffle request type.
#
# This IS the deterministic offline fallback used by llm_client.PlanningLLM
# when ANTHROPIC_API_KEY is unset (see planning/llm_client.py docstring).
# When a key IS set, decompose_goal() asks Claude for the same shape and
# only falls back to this if Claude's output fails Plan validation.
# ---------------------------------------------------------------------------
def _surge_plan_fallback(_user_prompt: str) -> dict:
    return {
        "goal": "ER surge reshuffle",
        "tasks": [
            {
                "id": "assess_incoming",
                "instruction": "Classify each incoming patient's urgency using the triage guidelines.",
                "depends_on": [],
                "kind": "reasoning",
                "mcp_tool": None,
            },
            {
                "id": "check_capacity",
                "instruction": "Fetch real free ICU beds and hospital capacity from the database.",
                "depends_on": [],
                "kind": "read",
                "mcp_tool": "get_available_icu_beds",
            },
            {
                "id": "rank_by_urgency",
                "instruction": "Order patients by urgency (RED before YELLOW before GREEN).",
                "depends_on": ["assess_incoming"],
                "kind": "reasoning",
                "mcp_tool": None,
            },
            {
                "id": "propose_assignment",
                "instruction": "Match the highest-urgency patients to real free ICU beds.",
                "depends_on": ["check_capacity", "rank_by_urgency"],
                "kind": "reasoning",
                "mcp_tool": None,
            },
            {
                # kind stays "reasoning" (not "read") even though its fallback
                # calls the real DB -- this node must set validated_assignments
                # in `state`, which is reasoning-branch behaviour; giving it
                # the same (kind, mcp_tool) pair as check_capacity would make
                # the generic executor treat it as a plain re-fetch and skip
                # that assignment. See execute_plan() dispatch below.
                "id": "validate_assignment",
                "instruction": "Re-check bed availability immediately before writing, to catch races.",
                "depends_on": ["propose_assignment"],
                "kind": "reasoning",
                "mcp_tool": None,
            },
            {
                "id": "apply_and_report",
                "instruction": "Register patients and occupy the validated beds; report the outcome.",
                "depends_on": ["validate_assignment"],
                "kind": "write",
                "mcp_tool": "manage_icu_bed",
            },
        ],
    }


def decompose_goal(goal: str, llm: PlanningLLM) -> Plan:
    payload = llm.structured(
        system=PLANNER_SYSTEM,
        user=f"Decompose this real hospital request into a DAG: {goal!r}",
        fallback_fn=_surge_plan_fallback,
        label="decomposition_first.plan",
    )
    payload = dict(payload)
    payload["goal"] = goal  # caller's goal stays authoritative even if paraphrased
    return Plan.model_validate(payload)


# ---------------------------------------------------------------------------
# Execution: reasoning nodes go through the LLM (with a domain-aware
# fallback); read/write nodes are executed for real against db_helpers.
# ---------------------------------------------------------------------------
def _reasoning_step(task: Task, goal: str, outputs: dict[str, Any], state: dict, llm: PlanningLLM) -> str:
    incoming = state["incoming_patients"]

    def fallback(_prompt: str) -> dict:
        if task.id == "assess_incoming":
            classified = []
            for p in incoming:
                text = p["diagnosis"].lower()
                if any(k in text for k in ("trauma", "cardiac", "arrest", "bleeding")):
                    severity = "RED"
                elif any(k in text for k in ("abdominal", "fever", "asthma")):
                    severity = "YELLOW"
                else:
                    severity = "GREEN"
                classified.append({**p, "severity": severity})
            state["classified_patients"] = classified
            return {"output": f"Classified {len(classified)} patients: "
                               + ", ".join(f"{c['name']}={c['severity']}" for c in classified)}

        if task.id == "rank_by_urgency":
            order = {"RED": 0, "YELLOW": 1, "GREEN": 2}
            ranked = sorted(state["classified_patients"], key=lambda p: (order[p["severity"]], -p["age"]))
            state["ranked_patients"] = ranked
            return {"output": "Order: " + ", ".join(f"{p['name']}({p['severity']})" for p in ranked)}

        if task.id == "propose_assignment":
            free_beds = state["free_beds_snapshot"]  # set by check_capacity node
            red_patients = [p for p in state["ranked_patients"] if p["severity"] == "RED"]
            assignments = []
            overflow = []
            for i, patient in enumerate(red_patients):
                if i < len(free_beds):
                    assignments.append({"patient": patient, "bed_id": free_beds[i]["bed_id"]})
                else:
                    overflow.append(patient)  # <-- decomposition-first has NO node for this
            state["assignments"] = assignments
            state["overflow"] = overflow
            msg = f"{len(assignments)} RED patient(s) matched to real free beds."
            if overflow:
                msg += (f" {len(overflow)} RED patient(s) LEFT WITHOUT A BED "
                        f"(no overflow/transfer task exists in this fixed plan): "
                        + ", ".join(p["name"] for p in overflow))
            return {"output": msg}

        if task.id == "validate_assignment":
            still_free = {b["bed_id"] for b in grounding.check_capacity().data["free_beds"]}
            valid = [a for a in state["assignments"] if a["bed_id"] in still_free]
            invalid = [a for a in state["assignments"] if a["bed_id"] not in still_free]
            state["validated_assignments"] = valid
            msg = f"{len(valid)}/{len(state['assignments'])} assignment(s) still valid at write time."
            if invalid:
                msg += f" REJECTED (bed taken meanwhile): {[a['bed_id'] for a in invalid]}"
            return {"output": msg}

        return {"output": f"(no domain fallback for {task.id})"}

    result = llm.structured(
        system="Execute one node of a validated hospital-DAG. Be concrete.",
        user=f"Goal: {goal}\nTask: {task.instruction}\nDependency outputs: {outputs}",
        fallback_fn=fallback,
        label=f"decomposition_first.{task.id}",
    )
    return result["output"]


def execute_plan(plan: Plan, llm: PlanningLLM, state: dict) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for batch in plan.execution_batches():
        for task_id in batch:
            task = plan.task(task_id)
            dep_outputs = {d: outputs[d] for d in task.depends_on}

            if task.kind == "read" and task.mcp_tool == "get_available_icu_beds":
                grounded = grounding.check_capacity()
                state["free_beds_snapshot"] = grounded.data["free_beds"]
                outputs[task_id] = grounded.summary

            elif task.kind == "write" and task.mcp_tool == "manage_icu_bed":
                applied = []
                state["created_patient_ids"] = []
                for a in state.get("validated_assignments", []):
                    p = a["patient"]
                    patient_id = db_register(p)
                    state["created_patient_ids"].append(patient_id)
                    r = grounding.assign_icu_bed(a["bed_id"], patient_id)
                    applied.append(r.summary)
                overflow_note = (
                    f" UNRESOLVED overflow: {[p['name'] for p in state.get('overflow', [])]}"
                    if state.get("overflow") else ""
                )
                outputs[task_id] = "; ".join(applied) + overflow_note

            else:  # reasoning
                outputs[task_id] = _reasoning_step(task, plan.goal, dep_outputs, state, llm)

    return outputs


def db_register(patient: dict) -> int:
    """Register a newly-arrived patient for real before occupying a bed."""
    from . import mcp_grounding as g
    return g.db.add_patient(
        {
            "name": patient["name"],
            "age": patient["age"],
            "gender": patient.get("gender", "Male"),
            "blood_type": patient.get("blood_type"),
            "diagnosis": patient["diagnosis"],
        }
    )


def final_output(plan: Plan, outputs: dict[str, str]) -> str:
    terminals = plan.terminal_tasks()
    if len(terminals) != 1:
        raise ValueError(f"Expected exactly one terminal synthesis task, found {terminals}")
    return outputs[terminals[0]]


def run_decomposition_first(goal: str, incoming_patients: list[dict], llm: PlanningLLM) -> dict:
    """Convenience entry point used by planning_eval/ and planning_agent.py."""
    t0 = time.time()
    plan = decompose_goal(goal, llm)
    state = {"incoming_patients": incoming_patients}
    outputs = execute_plan(plan, llm, state)
    return {
        "method": "decomposition_first",
        "plan": [t.model_dump() for t in plan.tasks],
        "topological_order": plan.topological_order(),
        "outputs": outputs,
        "final": final_output(plan, outputs),
        "unresolved_overflow": [p["name"] for p in state.get("overflow", [])],
        "created_patient_ids": state.get("created_patient_ids", []),
        "occupied_bed_ids": [a["bed_id"] for a in state.get("validated_assignments", [])],
        "llm_calls": llm.call_count,
        "input_tokens": llm.total_input_tokens,
        "output_tokens": llm.total_output_tokens,
        "latency_s": round(time.time() - t0, 4),
    }

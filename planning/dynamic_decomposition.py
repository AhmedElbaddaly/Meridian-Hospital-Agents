"""
planning/dynamic_decomposition.py
------------------------------------
Dynamic / interleaved decomposition: propose ONE next sub-task, execute
it, observe the real result, then decide the next sub-task -- so an
early surprise can reshape the rest of the plan. Forked from:
github.com/AmrSheta22/task_decomposition_and_planning
    (planning_lab/algorithms/dynamic_decomposition.py)

Same real request as decomposition.py (ER surge reshuffle), same real
grounding (planning/mcp_grounding.py, same DB). The only difference is
*when* the plan is committed.

This is where the lab's required divergence demo lives: after the
`check_capacity` action, this planner OBSERVES the real free-bed count.
If it is short of the number of RED patients already classified, it
inserts a task that decomposition-first's fixed plan never had:
`escalate_overflow` -- checking whether another Meridian hospital has
room, per the real `Hospitals` table. Decomposition-first cannot do this
because its plan was already fixed before that observation existed.
"""

from __future__ import annotations

import time
from typing import Any

from .llm_client import PlanningLLM
from . import mcp_grounding as grounding
from .decomposition import db_register


DECISION_SYSTEM = "You are an adaptive hospital planner. Decide the single best next step from real observations."


def _decide_next(goal: str, history: list[tuple[str, str]], state: dict, llm: PlanningLLM) -> dict:
    def fallback(_prompt: str) -> dict:
        done_ids = [h[0] for h in history]

        if "assess_incoming" not in done_ids:
            return {"done": False, "next_task": "assess_incoming"}

        if "check_capacity" not in done_ids:
            return {"done": False, "next_task": "check_capacity"}

        # --- the observation point where dynamic decomposition can diverge ---
        red_count = sum(1 for p in state["classified_patients"] if p["severity"] == "RED")
        free_count = len(state.get("free_beds_snapshot", []))
        shortfall = red_count - free_count

        if shortfall > 0 and "escalate_overflow" not in done_ids and "escalated" not in state:
            # A real surprise: decomposition-first's fixed plan has no node
            # for this. Dynamic decomposition reacts to it instead.
            return {"done": False, "next_task": "escalate_overflow"}

        if "rank_by_urgency" not in done_ids:
            return {"done": False, "next_task": "rank_by_urgency"}

        if "propose_assignment" not in done_ids:
            return {"done": False, "next_task": "propose_assignment"}

        if "validate_assignment" not in done_ids:
            return {"done": False, "next_task": "validate_assignment"}

        if "apply_and_report" not in done_ids:
            return {"done": False, "next_task": "apply_and_report"}

        return {"done": True, "next_task": ""}

    obs = "\n".join(f"{t}: {r}" for t, r in history) or "None"
    return llm.structured(
        system=DECISION_SYSTEM,
        user=f"Goal: {goal}\nObservations so far:\n{obs}",
        fallback_fn=fallback,
        label="dynamic.decide_next",
    )


def _execute_task(task_id: str, goal: str, state: dict, llm: PlanningLLM) -> str:
    if task_id == "assess_incoming":
        classified = []
        for p in state["incoming_patients"]:
            text = p["diagnosis"].lower()
            if any(k in text for k in ("trauma", "cardiac", "arrest", "bleeding")):
                severity = "RED"
            elif any(k in text for k in ("abdominal", "fever", "asthma")):
                severity = "YELLOW"
            else:
                severity = "GREEN"
            classified.append({**p, "severity": severity})
        state["classified_patients"] = classified
        return "Classified: " + ", ".join(f"{c['name']}={c['severity']}" for c in classified)

    if task_id == "check_capacity":
        grounded = grounding.check_capacity()
        state["free_beds_snapshot"] = grounded.data["free_beds"]
        return grounded.summary

    if task_id == "escalate_overflow":
        # Real check against the Hospitals table -- this branch literally
        # does not exist in decomposition.py's fixed plan.
        from . import mcp_grounding as g
        other = g.db.get_hospital_info(3)  # International Medical Center -- highest real capacity in seed data
        state["escalated"] = True
        beds = other.get("available_icu_beds", 0) if other else 0
        if beds > 0:
            state["overflow_route"] = f"transfer to {other['hospital_name']} ({beds} free ICU beds recorded)"
        else:
            state["overflow_route"] = "no partner hospital capacity found -- hold overflow patient(s) in ER bay, re-check in 15 min"
        return f"Overflow check: {state['overflow_route']}"

    if task_id == "rank_by_urgency":
        order = {"RED": 0, "YELLOW": 1, "GREEN": 2}
        ranked = sorted(state["classified_patients"], key=lambda p: (order[p["severity"]], -p["age"]))
        state["ranked_patients"] = ranked
        return "Order: " + ", ".join(f"{p['name']}({p['severity']})" for p in ranked)

    if task_id == "propose_assignment":
        free_beds = state["free_beds_snapshot"]
        red_patients = [p for p in state["ranked_patients"] if p["severity"] == "RED"]
        assignments, overflow = [], []
        for i, patient in enumerate(red_patients):
            if i < len(free_beds):
                assignments.append({"patient": patient, "bed_id": free_beds[i]["bed_id"]})
            else:
                overflow.append(patient)
        state["assignments"] = assignments
        state["overflow"] = overflow
        msg = f"{len(assignments)} RED patient(s) matched to real free beds."
        if overflow:
            route = state.get("overflow_route", "no route determined")
            msg += f" {len(overflow)} routed via escalation plan: {route}"
        return msg

    if task_id == "validate_assignment":
        still_free = {b["bed_id"] for b in grounding.check_capacity().data["free_beds"]}
        valid = [a for a in state["assignments"] if a["bed_id"] in still_free]
        state["validated_assignments"] = valid
        return f"{len(valid)}/{len(state['assignments'])} assignment(s) still valid at write time."

    if task_id == "apply_and_report":
        applied = []
        state["created_patient_ids"] = []
        for a in state.get("validated_assignments", []):
            patient_id = db_register(a["patient"])
            state["created_patient_ids"].append(patient_id)
            r = grounding.assign_icu_bed(a["bed_id"], patient_id)
            applied.append(r.summary)
        note = f" Escalated overflow handled via: {state['overflow_route']}" if state.get("escalated") else ""
        return "; ".join(applied) + note

    return f"(unknown task {task_id})"


def dynamic_decomposition(goal: str, incoming_patients: list[dict], llm: PlanningLLM, max_steps: int = 8) -> dict:
    t0 = time.time()
    state: dict[str, Any] = {"incoming_patients": incoming_patients}
    history: list[tuple[str, str]] = []

    for _ in range(max_steps):
        decision = _decide_next(goal, history, state, llm)
        if decision["done"]:
            break
        task_id = decision["next_task"]
        result = _execute_task(task_id, goal, state, llm)
        history.append((task_id, result))

    return {
        "method": "dynamic_decomposition",
        "executed_order": [t for t, _ in history],
        "outputs": dict(history),
        "final": history[-1][1] if history else "",
        "diverged_with_escalation": "escalate_overflow" in [t for t, _ in history],
        "created_patient_ids": state.get("created_patient_ids", []),
        "occupied_bed_ids": [a["bed_id"] for a in state.get("validated_assignments", [])],
        "llm_calls": llm.call_count,
        "input_tokens": llm.total_input_tokens,
        "output_tokens": llm.total_output_tokens,
        "latency_s": round(time.time() - t0, 4),
    }

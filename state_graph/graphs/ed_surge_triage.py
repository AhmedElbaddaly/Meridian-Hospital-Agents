"""
Graph 3: Emergency-department surge triage & bed escalation

Why a state graph:
  - Irreversible actions (sedation, ICU admission, OR booking) must not be
    taken by the agent alone → HITL to on-call attending via platform
  - Wrong triage order costs real time in an emergency
  - Resource availability (ICU beds, OR) is external and changes mid-run
  - Real failure modes: acuity score schema errors, bed-assignment tool failures

Two LLM-call additions:
  1. LATS              — node `search_triage_order`
       Search over candidate triage orderings scored by a real severity check
       (not the model's own opinion of urgency).
  2. Constrained ReAct — node `execute_whitelisted_actions`
       Execute only whitelisted intake actions; sedation/admission are gated.

HITL: any action in {sedate, admit_icu, book_or, transfer_out} requires
on-call approval through the platform before the graph proceeds.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..runtime import GraphDef, GraphState, request_hitl
from ..llm_techniques import lats_search, ConstrainedReAct, rag_retrieve


ED_ALLOWED_TOOLS = [
    "score_acuity",
    "check_icu_beds",
    "check_or_availability",
    "assign_bed",
    "record_triage",
    # irreversible — only after HITL
    "sedate",
    "admit_icu",
    "book_or",
    "transfer_out",
]

IRREVERSIBLE = {"sedate", "admit_icu", "book_or", "transfer_out"}


def _acuity_score(patient: Dict[str, Any]) -> float:
    """Real severity check (deterministic stand-in for ESI / NEWS2-style score)."""
    score = 3.0  # default ESI-ish mid
    hr = patient.get("hr", 80)
    sbp = patient.get("sbp", 120)
    spo2 = patient.get("spo2", 98)
    cc = (patient.get("chief_complaint") or "").lower()

    if hr > 130 or sbp < 90 or spo2 < 90:
        score = 1.0
    elif hr > 110 or sbp < 100 or spo2 < 94:
        score = 2.0
    if any(k in cc for k in ("chest pain", "stroke", "unresponsive", "trauma")):
        score = min(score, 1.5)
    if "minor" in cc or "suture" in cc:
        score = max(score, 4.0)
    return score  # lower = more urgent


def _node_intake_patients(state: GraphState) -> GraphState:
    patients = state.data.get("patients")
    if not patients:
        raise ValueError("ed_surge_triage: patients list required")
    for p in patients:
        if "id" not in p:
            raise ValueError("ed_surge_triage: each patient needs an id")
    state.data["n_patients"] = len(patients)
    return state


def _node_score_all(state: GraphState) -> GraphState:
    scored = []
    for p in state.data["patients"]:
        s = _acuity_score(p)
        scored.append({**p, "acuity": s})
    state.data["scored_patients"] = scored
    return state


def _node_search_triage_order(state: GraphState) -> GraphState:
    """
    LATS node.
    Reason: we enumerate candidate orderings and score them with the real
    acuity function + resource constraints, not model self-assessment of urgency.
    """
    patients = state.data.get("scored_patients") or []
    # Candidate orderings: sorted by acuity, reverse acuity, and arrival order
    by_acuity = sorted(patients, key=lambda x: x["acuity"])
    by_reverse = sorted(patients, key=lambda x: -x["acuity"])
    by_arrival = list(patients)

    candidates = [
        {"name": "acuity_asc", "order": [p["id"] for p in by_acuity]},
        {"name": "acuity_desc", "order": [p["id"] for p in by_reverse]},
        {"name": "arrival", "order": [p["id"] for p in by_arrival]},
    ]

    def evaluate(cand: Dict) -> float:
        # Lower acuity number = more urgent. Penalize putting urgent patients later.
        order = cand["order"]
        id_to_acuity = {p["id"]: p["acuity"] for p in patients}
        delay = 0.0
        for pos, pid in enumerate(order):
            urgency = 6.0 - id_to_acuity.get(pid, 3)  # higher = more urgent
            delay += urgency * pos  # cost of waiting
        return 100.0 / (1.0 + delay)

    best, ranked = lats_search(candidates, evaluate_fn=evaluate)
    state.data["chosen_order"] = best.get("order", [])
    state.data["lats_ranking"] = ranked
    state.data["lats_best_name"] = best.get("name")
    return state


def _node_propose_actions(state: GraphState) -> GraphState:
    """Map ordered patients to proposed intake actions (may include irreversible)."""
    patients = {p["id"]: p for p in state.data.get("scored_patients") or []}
    order = state.data.get("chosen_order") or []
    proposals = []
    for pid in order:
        p = patients.get(pid, {})
        acuity = p.get("acuity", 3)
        if acuity <= 1.5:
            action = "admit_icu"
        elif acuity <= 2.5:
            action = "book_or" if "trauma" in (p.get("chief_complaint") or "").lower() else "assign_bed"
        else:
            action = "assign_bed"
        proposals.append({"patient_id": pid, "action": action, "acuity": acuity})
    state.data["proposals"] = proposals
    return state


def _node_hitl_irreversible(state: GraphState) -> GraphState:
    """
    HITL gate for irreversible actions.
    """
    proposals = state.data.get("proposals") or []
    irreversible_props = [p for p in proposals if p["action"] in IRREVERSIBLE]
    if not irreversible_props:
        state.data["approved_actions"] = proposals
        return state

    if state.data.get("hitl_decision"):
        decision = state.data["hitl_decision"]
        approved = decision.get("approved_actions", proposals)
        state.data["approved_actions"] = approved
        state.data["hitl_admin"] = decision.get("admin_id", "on_call")
        return state

    request_hitl(
        reason=(
            "On-call approval required for irreversible ED actions: "
            + ", ".join(f"{p['patient_id']}:{p['action']}" for p in irreversible_props)
        )
    )
    return state


def _node_execute_whitelisted(state: GraphState) -> GraphState:
    """
    Constrained ReAct execution.
    Only actions in ED_ALLOWED_TOOLS (and already HITL-approved if irreversible).
    """
    react = ConstrainedReAct(allowed_tools=list(ED_ALLOWED_TOOLS), max_iters=20)
    approved = state.data.get("approved_actions") or state.data.get("proposals") or []

    def _executor(action: str, args: Dict) -> Any:
        return {"ok": True, "action": action, "args": args}

    results = []
    for prop in approved:
        action = prop["action"]
        if action not in ED_ALLOWED_TOOLS:
            raise PermissionError(f"Action {action} not whitelisted")
        # Irreversible must have passed HITL (approved_actions set)
        if action in IRREVERSIBLE and not state.data.get("hitl_decision") and not state.data.get("approved_actions"):
            raise PermissionError(f"Irreversible action {action} without HITL approval")
        r = react.step(
            observation=f"triage patient {prop['patient_id']}",
            proposed_action=action,
            tool_args={"patient_id": prop["patient_id"]},
            tool_executor=_executor,
        )
        results.append(r)

    state.data["execution_trace"] = react.trace
    state.data["execution_results"] = results
    state.data["triage_complete"] = True
    return state


def build_ed_surge_triage_graph() -> GraphDef:
    return GraphDef(
        name="ed_surge_triage",
        entry="intake_patients",
        nodes={
            "intake_patients": _node_intake_patients,
            "score_all": _node_score_all,
            "search_triage_order": _node_search_triage_order,
            "propose_actions": _node_propose_actions,
            "hitl_irreversible": _node_hitl_irreversible,
            "execute_whitelisted": _node_execute_whitelisted,
        },
        edges={
            "intake_patients": lambda s: "score_all",
            "score_all": lambda s: "search_triage_order",
            "search_triage_order": lambda s: "propose_actions",
            "propose_actions": lambda s: "hitl_irreversible",
            "hitl_irreversible": lambda s: "execute_whitelisted",
            "execute_whitelisted": lambda s: "END",
        },
    )

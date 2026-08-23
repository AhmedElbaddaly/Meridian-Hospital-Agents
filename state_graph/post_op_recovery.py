"""
Graph 1: Multi-day post-operative recovery coordination

Why a state graph (not a single pass / DAG):
  - Spans days: waits on lab results that arrive on their own schedule
  - Real branch outside the model: lab webhook / timeout
  - Real cost to losing progress: re-collecting vitals and re-ordering labs
  - Physician sign-off before plan changes is a genuine HITL node

Two LLM-call additions (placed inside specific nodes for a reason):
  1. Task decomposition  — node `plan_recovery_sequence`
       Builds the ordered visit/check sequence from the high-level recovery goal.
  2. RAG                — node `review_against_protocol`
       Pulls the clinic's chronic / post-op care protocols so the comparison
       is grounded in real hospital policy, not model memory.

HITL condition: any recommendation to advance phase, escalate to ICU, or
discharge requires physician sign-off when confidence < 0.75 OR action is
irreversible (discharge / ICU transfer).

Failure ticket examples: lab schema validation failure, missing patient_id,
tool call error when writing vitals via MCP.
"""

from __future__ import annotations

from typing import Any, Dict

from ..runtime import GraphDef, GraphState, request_hitl
from ..llm_techniques import task_decomposition, rag_retrieve


def _node_intake(state: GraphState) -> GraphState:
    """Collect patient + procedure context. Requires patient_id."""
    data = state.data
    if not data.get("patient_id"):
        raise ValueError("post_op_recovery: patient_id is required (schema validation)")
    data.setdefault("procedure", "unspecified")
    data.setdefault("day", 0)
    data.setdefault("phase", "immediate")
    data["intake_complete"] = True
    return state


def _node_plan_recovery_sequence(state: GraphState) -> GraphState:
    """
    TASK DECOMPOSITION node.
    Reason: the recovery goal is multi-day and multi-check; we need an explicit
    ordered plan before we start executing, not ad-hoc tool calls.
    """
    goal = (
        f"Coordinate post-op recovery for patient {state.data.get('patient_id')} "
        f"after {state.data.get('procedure')}"
    )
    plan = task_decomposition(goal, context=state.data)
    state.data["recovery_plan"] = plan
    state.data["plan_index"] = 0
    return state


def _node_collect_vitals(state: GraphState) -> GraphState:
    """Simulate / accept vitals. In production this calls an MCP write tool."""
    vitals = state.data.get("vitals") or {
        "hr": 88,
        "sbp": 118,
        "dbp": 72,
        "temp_c": 37.1,
        "pain_score": 3,
    }
    state.data["vitals"] = vitals
    state.data["vitals_collected"] = True
    return state


def _node_order_labs(state: GraphState) -> GraphState:
    state.data["labs_ordered"] = True
    state.data["labs_status"] = "pending"
    return state


def _node_await_lab_results(state: GraphState) -> GraphState:
    """
    Genuine waiting state. Transitions only when labs arrive or timeout fires.
    In a live deployment a lab-integration webhook would set labs_status=ready.
    """
    status = state.data.get("labs_status", "pending")
    if status == "pending" and not state.data.get("force_labs_ready"):
        # Stay here — runtime will checkpoint; external event or admin can advance
        state.data["waiting_for"] = "lab_results"
        # For demo/deterministic runs, allow a flag to simulate arrival
        if state.data.get("simulate_lab_arrival"):
            state.data["labs_status"] = "ready"
            state.data["lab_results"] = state.data.get("lab_results") or {
                "hgb": 11.2,
                "wbc": 9.1,
                "creatinine": 0.9,
                "k": 4.0,
            }
            state.data.pop("waiting_for", None)
        else:
            # Keep node identity so edge can loop back
            state.data["labs_still_pending"] = True
    else:
        state.data["labs_status"] = "ready"
        state.data.setdefault(
            "lab_results",
            {"hgb": 11.2, "wbc": 9.1, "creatinine": 0.9, "k": 4.0},
        )
        state.data.pop("labs_still_pending", None)
        state.data.pop("waiting_for", None)
    return state


def _node_review_against_protocol(state: GraphState) -> GraphState:
    """
    RAG node.
    Reason: comparison must be grounded in the hospital's post-op protocol,
    not the model's parametric knowledge of generic guidelines.
    """
    query = (
        f"post-operative recovery protocol phase {state.data.get('phase')} "
        f"vitals {state.data.get('vitals')} labs {state.data.get('lab_results')}"
    )
    docs = rag_retrieve(query, top_k=2)
    state.data["protocol_snippets"] = docs

    vitals = state.data.get("vitals") or {}
    labs = state.data.get("lab_results") or {}
    concerns = []
    if vitals.get("hr", 0) > 120:
        concerns.append("tachycardia")
    if vitals.get("sbp", 999) < 90:
        concerns.append("hypotension")
    if vitals.get("pain_score", 0) > 7:
        concerns.append("uncontrolled_pain")
    if labs.get("hgb", 99) < 8:
        concerns.append("severe_anemia")

    if concerns:
        state.data["recommendation"] = "escalate"
        state.data["concerns"] = concerns
        state.data["confidence"] = 0.55
    else:
        state.data["recommendation"] = "advance_phase"
        state.data["concerns"] = []
        state.data["confidence"] = 0.82
    return state


def _node_physician_signoff(state: GraphState) -> GraphState:
    """
    HITL node.
    Conditions that must not let the agent decide alone:
      - recommendation in {escalate, discharge, icu_transfer}
      - confidence < 0.75
    """
    rec = state.data.get("recommendation", "hold")
    conf = float(state.data.get("confidence", 0.5))
    irreversible = rec in ("escalate", "discharge", "icu_transfer", "advance_phase")

    if irreversible or conf < 0.75:
        # If admin already decided on a previous resume, honour it
        if state.data.get("hitl_decision"):
            decision = state.data["hitl_decision"]
            state.data["final_plan"] = decision.get("plan", rec)
            state.data["signed_off_by"] = decision.get("admin_id", "admin")
            return state
        request_hitl(
            reason=(
                f"Physician sign-off required: recommendation={rec}, "
                f"confidence={conf:.2f}, concerns={state.data.get('concerns')}"
            )
        )
    state.data["final_plan"] = rec
    return state


def _node_apply_plan(state: GraphState) -> GraphState:
    plan = state.data.get("final_plan") or state.data.get("recommendation")
    state.data["applied_plan"] = plan
    if plan == "advance_phase":
        state.data["phase"] = "intermediate"
        state.data["day"] = int(state.data.get("day", 0)) + 1
    elif plan == "escalate":
        state.data["phase"] = "escalated"
    state.data["recovery_complete"] = plan in ("advance_phase", "discharge", "hold")
    return state


def _edge_after_await(state: GraphState) -> str:
    if state.data.get("labs_still_pending"):
        return "await_lab_results"  # loop / wait
    return "review_against_protocol"


def _edge_after_signoff(state: GraphState) -> str:
    if state.status == "waiting_hitl":
        return "physician_signoff"
    return "apply_plan"


def build_post_op_recovery_graph() -> GraphDef:
    return GraphDef(
        name="post_op_recovery",
        entry="intake",
        nodes={
            "intake": _node_intake,
            "plan_recovery_sequence": _node_plan_recovery_sequence,
            "collect_vitals": _node_collect_vitals,
            "order_labs": _node_order_labs,
            "await_lab_results": _node_await_lab_results,
            "review_against_protocol": _node_review_against_protocol,
            "physician_signoff": _node_physician_signoff,
            "apply_plan": _node_apply_plan,
        },
        edges={
            "intake": lambda s: "plan_recovery_sequence",
            "plan_recovery_sequence": lambda s: "collect_vitals",
            "collect_vitals": lambda s: "order_labs",
            "order_labs": lambda s: "await_lab_results",
            "await_lab_results": _edge_after_await,
            "review_against_protocol": lambda s: "physician_signoff",
            "physician_signoff": _edge_after_signoff,
            "apply_plan": lambda s: "END",
        },
    )

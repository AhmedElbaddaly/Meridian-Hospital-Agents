"""
Graph 2: Elective-procedure insurance pre-authorization & appeal

Why a state graph:
  - Waits on an external insurer's response (may take days)
  - Can be rejected and need a reasoned appeal (branch outside the model)
  - Wrong resubmission wastes a real claim window
  - Malformed insurer response → failure ticket, not silent failure

Two LLM-call additions:
  1. Constrained ReAct — node `submit_or_appeal`
       Fill and submit only the correct insurer forms / whitelisted write tools.
  2. Tree of Thoughts  — node `choose_appeal_strategy`
       Choose which appeal argument to lead with after a denial.

HITL: appeal filing when estimated charge > hospital threshold, or when
appeal strategy confidence is low.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..runtime import GraphDef, GraphState, request_hitl
from ..llm_techniques import tree_of_thoughts, ConstrainedReAct


# Whitelist of MCP-style tools this graph may call
INSURANCE_ALLOWED_TOOLS = [
    "gather_clinical_packet",
    "select_insurer_forms",
    "submit_preauth",
    "file_appeal",
    "record_decision",
]


def _node_gather_packet(state: GraphState) -> GraphState:
    if not state.data.get("patient_id"):
        raise ValueError("insurance_auth: patient_id required")
    if not state.data.get("cpt_code"):
        raise ValueError("insurance_auth: cpt_code required for preauth")
    state.data.setdefault("icd10", "M17.11")
    state.data.setdefault("estimated_charge", 12000)
    state.data["packet"] = {
        "patient_id": state.data["patient_id"],
        "cpt": state.data["cpt_code"],
        "icd10": state.data["icd10"],
        "notes": state.data.get("clinical_notes", "Elective procedure indicated."),
    }
    return state


def _node_submit_preauth(state: GraphState) -> GraphState:
    """
    Constrained ReAct submission path.
    Only whitelisted actions may run; anything else becomes a ticket.
    """
    react = ConstrainedReAct(allowed_tools=INSURANCE_ALLOWED_TOOLS, max_iters=4)
    plan = [
        {"action": "select_insurer_forms", "args": {"payer": state.data.get("payer", "generic")}},
        {"action": "submit_preauth", "args": state.data.get("packet", {})},
    ]

    def _executor(action: str, args: Dict) -> Any:
        # Offline stand-in for MCP tool calls
        if action == "submit_preauth":
            # Simulate external response if provided, else pending
            if state.data.get("simulate_insurer_response"):
                return state.data["simulate_insurer_response"]
            return {"status": "pending"}
        return {"ok": True, "action": action}

    results = react.run_plan(plan, tool_executor=_executor)
    state.data["react_trace"] = react.trace
    last = results[-1]["output"] if results else {}
    state.data["preauth_status"] = last.get("status", "pending")
    if state.data["preauth_status"] == "denied":
        state.data["denial_reason"] = last.get("reason", "medical necessity not established")
    return state


def _node_await_insurer(state: GraphState) -> GraphState:
    status = state.data.get("preauth_status", "pending")
    if status == "pending" and not state.data.get("force_insurer_ready"):
        if state.data.get("simulate_insurer_response"):
            resp = state.data["simulate_insurer_response"]
            state.data["preauth_status"] = resp.get("status", "approved")
            if state.data["preauth_status"] == "denied":
                state.data["denial_reason"] = resp.get("reason", "not medically necessary")
        else:
            state.data["waiting_for"] = "insurer_response"
            state.data["insurer_still_pending"] = True
            return state
    state.data.pop("insurer_still_pending", None)
    state.data.pop("waiting_for", None)
    return state


def _node_choose_appeal_strategy(state: GraphState) -> GraphState:
    """
    Tree of Thoughts node.
    Reason: after denial we must choose among distinct appeal arguments;
    ToT scores them against the prior denial reason and clinical packet.
    """
    if state.data.get("preauth_status") != "denied":
        state.data["appeal_needed"] = False
        return state

    candidates = [
        "Lead with specialty society guideline citation and medical necessity narrative",
        "Lead with cost-effectiveness and alternative-treatment failure history",
        "Lead with imaging / objective findings that were under-weighted in denial",
        "Request peer-to-peer review with treating surgeon",
    ]
    context = {
        "prior_denial_reason": state.data.get("denial_reason", ""),
        "cpt": state.data.get("cpt_code"),
    }
    best, scored = tree_of_thoughts(
        problem="Select strongest insurance appeal strategy",
        candidates=candidates,
        context=context,
    )
    state.data["appeal_strategy"] = best
    state.data["appeal_scores"] = scored
    state.data["appeal_needed"] = True
    state.data["appeal_confidence"] = scored[0]["score"] if scored else 0.5
    return state


def _node_file_appeal_or_finish(state: GraphState) -> GraphState:
    if not state.data.get("appeal_needed"):
        state.data["final_status"] = state.data.get("preauth_status", "approved")
        return state

    conf = float(state.data.get("appeal_confidence", 0.5))
    charge = float(state.data.get("estimated_charge", 0))
    # HITL when charge above threshold or low confidence
    if charge >= 10000 or conf < 0.6:
        if state.data.get("hitl_decision"):
            decision = state.data["hitl_decision"]
            if decision.get("approve_appeal"):
                state.data["appeal_filed"] = True
                state.data["final_status"] = "appeal_submitted"
            else:
                state.data["appeal_filed"] = False
                state.data["final_status"] = "denied_no_appeal"
            return state
        request_hitl(
            reason=(
                f"Appeal filing requires admin approval: charge={charge}, "
                f"confidence={conf:.2f}, strategy={state.data.get('appeal_strategy')}"
            )
        )

    # Constrained ReAct for the actual filing
    react = ConstrainedReAct(allowed_tools=INSURANCE_ALLOWED_TOOLS)
    react.step(
        observation="filing appeal with chosen strategy",
        proposed_action="file_appeal",
        tool_args={
            "strategy": state.data.get("appeal_strategy"),
            "packet": state.data.get("packet"),
        },
    )
    state.data["react_trace_appeal"] = react.trace
    state.data["appeal_filed"] = True
    state.data["final_status"] = "appeal_submitted"
    return state


def _edge_after_await(state: GraphState) -> str:
    if state.data.get("insurer_still_pending"):
        return "await_insurer"
    if state.data.get("preauth_status") == "denied":
        return "choose_appeal_strategy"
    return "file_appeal_or_finish"


def build_insurance_auth_graph() -> GraphDef:
    return GraphDef(
        name="insurance_auth",
        entry="gather_packet",
        nodes={
            "gather_packet": _node_gather_packet,
            "submit_preauth": _node_submit_preauth,
            "await_insurer": _node_await_insurer,
            "choose_appeal_strategy": _node_choose_appeal_strategy,
            "file_appeal_or_finish": _node_file_appeal_or_finish,
        },
        edges={
            "gather_packet": lambda s: "submit_preauth",
            "submit_preauth": lambda s: "await_insurer",
            "await_insurer": _edge_after_await,
            "choose_appeal_strategy": lambda s: "file_appeal_or_finish",
            "file_appeal_or_finish": lambda s: "END",
        },
    )

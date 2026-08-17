"""
planning/plan_and_solve.py
----------------------------
Plan-and-Solve: one explicit plan phase, then one execute phase, single
pass, no branching, no search. Forked from:
github.com/AmrSheta22/task_decomposition_and_planning
    (planning_lab/algorithms/plan_and_solve.py)

The toolkit version is a single LangChain `llm.invoke(...)` call against
ChatMistralAI. We keep the exact two-phase Plan-and-Solve prompting idea
(Wang et al., ACL 2023 -- "first devise a plan, then carry it out step by
step") but:
  - swap ChatMistralAI for planning/llm_client.PlanningLLM (team's shared
    provider wrapper, per llm_client.py's own docstring), so call_count /
    total_tokens bookkeeping happens automatically for the comparison
    table -- the toolkit version has none of that.
  - split the single free-text call into a *structured* plan call and a
    structured solve call, because our caller (planning/router.py /
    agent/planning_agent.py) needs a machine-readable `output` field to
    feed into the next DAG node, not a paragraph to re-parse.
  - add a real domain fallback (see `_assess_incoming_fallback` below)
    instead of the toolkit's bare "no API key -> can't run" behaviour,
    so this module is testable and demoable offline like every other
    planning/ module (team convention set by llm_client.py).

Why Plan-and-Solve fits `assess_incoming` and not the other reasoning
nodes (see planning/router.py for the full rationale table):
  - triage classification against a fixed, published rule set (RED /
    YELLOW / GREEN, see mcp_grounding.TRIAGE_GUIDELINES) has exactly one
    correct answer per patient -- there is nothing to branch on and
    nothing a second candidate would improve. Tree of Thoughts here would
    just pay for 2-4x the calls to re-derive the same classification.
  - it is cheap to get wrong in a way Self-Refine (Person 3) can catch
    on one revision -- it does not need LATS's expensive external-search
    machinery, and a wrong triage call has no "several valid outcomes"
    the way bed-ranking or bed-assignment do.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .llm_client import PlanningLLM

PLAN_SYSTEM = (
    "You use Plan-and-Solve prompting for a hospital ER-triage sub-task. "
    "First understand the problem and devise a short plan. Do not solve yet."
)
SOLVE_SYSTEM = (
    "You use Plan-and-Solve prompting for a hospital ER-triage sub-task. "
    "Carry out the plan step by step. Check each assumption against the "
    "triage rule set given to you before committing to an answer."
)


@dataclass
class PlanAndSolveResult:
    task_id: str
    plan: str
    output: str
    llm_calls: int
    input_tokens: int
    output_tokens: int
    latency_s: float
    state_patch: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Deterministic offline fallback for the ONE sub-task this module is routed
# to today (assess_incoming). Same convention as decomposition.py's
# _surge_plan_fallback: not a random stub, it encodes the real triage rule
# (mcp_grounding.TRIAGE_GUIDELINES) so the offline demo is meaningful.
# ---------------------------------------------------------------------------
def _assess_incoming_fallback(incoming_patients: list[dict]) -> tuple[str, dict]:
    plan_text = (
        "1) Read each patient's diagnosis text. "
        "2) Match against RED/YELLOW/GREEN keyword rules. "
        "3) Assign severity; keep original patient order."
    )
    classified = []
    for p in incoming_patients:
        text = p["diagnosis"].lower()
        if any(k in text for k in ("trauma", "cardiac", "arrest", "bleeding")):
            severity = "RED"
        elif any(k in text for k in ("abdominal", "fever", "asthma")):
            severity = "YELLOW"
        else:
            severity = "GREEN"
        classified.append({**p, "severity": severity})
    output_text = "Classified " + ", ".join(f"{c['name']}={c['severity']}" for c in classified)
    return plan_text, {"classified_patients": classified, "output": output_text}


def plan_and_solve(
    task_id: str,
    goal: str,
    instruction: str,
    llm: PlanningLLM,
    state: dict,
) -> PlanAndSolveResult:
    """Two-phase Plan-and-Solve: an explicit PLAN call, then one SOLVE call.

    Only `assess_incoming` has a wired domain fallback today (that is the
    node planning/router.py sends here); calling this with any other
    task_id in offline mode raises, the same way decomposition.py's
    fallback functions are single-purpose rather than silently guessing.
    """
    t0 = time.time()
    calls_before = llm.call_count
    in_tokens_before = llm.total_input_tokens
    out_tokens_before = llm.total_output_tokens

    def plan_fallback(_prompt: str) -> dict:
        if task_id != "assess_incoming":
            raise NotImplementedError(
                f"plan_and_solve has no offline fallback for task_id={task_id!r}; "
                "route it elsewhere or run online (ANTHROPIC_API_KEY set)."
            )
        plan_text, _ = _assess_incoming_fallback(state["incoming_patients"])
        return {"plan": plan_text}

    plan_payload = llm.structured(
        system=PLAN_SYSTEM,
        user=f"Goal: {goal}\nSub-task: {instruction}\nDevise a short numbered plan only.",
        fallback_fn=plan_fallback,
        label=f"plan_and_solve.{task_id}.plan",
    )
    plan_text = plan_payload["plan"]

    def solve_fallback(_prompt: str) -> dict:
        _, result = _assess_incoming_fallback(state["incoming_patients"])
        return result

    solve_payload = llm.structured(
        system=SOLVE_SYSTEM,
        user=(
            f"Goal: {goal}\nSub-task: {instruction}\nPlan:\n{plan_text}\n"
            "Execute the plan now. Return the classification result."
        ),
        fallback_fn=solve_fallback,
        label=f"plan_and_solve.{task_id}.solve",
    )

    state_patch = {k: v for k, v in solve_payload.items() if k != "output"}
    if state_patch:
        state.update(state_patch)

    return PlanAndSolveResult(
        task_id=task_id,
        plan=plan_text,
        output=solve_payload["output"],
        llm_calls=llm.call_count - calls_before,
        input_tokens=llm.total_input_tokens - in_tokens_before,
        output_tokens=llm.total_output_tokens - out_tokens_before,
        latency_s=round(time.time() - t0, 4),
        state_patch=state_patch,
    )

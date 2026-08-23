"""
Reusable LLM-call techniques used inside graph nodes.

Each graph must integrate TWO of:
  - task_decomposition
  - tree_of_thoughts / LATS
  - constrained_react
  - rag

These are placed inside specific nodes for a reason tied to that node's job.
Offline deterministic fallbacks exist so demos work without an API key.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Task Decomposition
# ---------------------------------------------------------------------------

def task_decomposition(
    goal: str,
    context: Dict[str, Any],
    max_steps: int = 6,
) -> List[Dict[str, str]]:
    """
    Break a high-level clinical / operational goal into an ordered sequence
    of concrete sub-tasks. Used when a node must plan a multi-visit or
    multi-form workflow before acting.
    """
    # Deterministic offline plan keyed by goal keywords (no API required).
    goal_l = goal.lower()
    if "recovery" in goal_l or "post-op" in goal_l or "post_op" in goal_l:
        steps = [
            {"id": "t1", "action": "collect_baseline_vitals", "desc": "Record post-op vitals and pain score"},
            {"id": "t2", "action": "order_labs", "desc": "Order CBC, chemistry, relevant markers"},
            {"id": "t3", "action": "await_lab_results", "desc": "Wait for lab integration results"},
            {"id": "t4", "action": "review_against_protocol", "desc": "Compare results to recovery protocol"},
            {"id": "t5", "action": "decide_next_phase", "desc": "Advance, hold, or escalate recovery phase"},
            {"id": "t6", "action": "physician_signoff", "desc": "Obtain physician sign-off if plan changes"},
        ]
    elif "insurance" in goal_l or "preauth" in goal_l or "authorization" in goal_l:
        steps = [
            {"id": "t1", "action": "gather_clinical_packet", "desc": "Assemble diagnosis, procedure codes, notes"},
            {"id": "t2", "action": "select_insurer_forms", "desc": "Choose correct payer-specific forms"},
            {"id": "t3", "action": "submit_preauth", "desc": "Submit pre-authorization request"},
            {"id": "t4", "action": "await_insurer_response", "desc": "Wait for external insurer decision"},
            {"id": "t5", "action": "handle_decision", "desc": "Approve path or prepare reasoned appeal"},
        ]
    elif "triage" in goal_l or "surge" in goal_l or "ed" in goal_l:
        steps = [
            {"id": "t1", "action": "score_acuity", "desc": "Run real acuity scoring against vitals/chief complaint"},
            {"id": "t2", "action": "enumerate_orderings", "desc": "Enumerate candidate triage orderings"},
            {"id": "t3", "action": "search_best_order", "desc": "LATS search over orderings vs severity"},
            {"id": "t4", "action": "propose_actions", "desc": "Propose only whitelisted intake actions"},
            {"id": "t5", "action": "hitl_irreversible", "desc": "Escalate sedation/admission/ICU to on-call"},
        ]
    else:
        steps = [
            {"id": f"t{i}", "action": f"step_{i}", "desc": f"Generic sub-task {i} for: {goal}"}
            for i in range(1, min(max_steps, 4) + 1)
        ]
    return steps[:max_steps]


# ---------------------------------------------------------------------------
# 2. Tree of Thoughts / LATS (Lightweight search over candidates)
# ---------------------------------------------------------------------------

def tree_of_thoughts(
    problem: str,
    candidates: List[str],
    scorer: Optional[Callable[[str, Dict], float]] = None,
    context: Optional[Dict[str, Any]] = None,
    beam: int = 3,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate / evaluate a small set of candidate thoughts and pick the best.
    Used for appeal-strategy selection and triage-ordering search.
    Returns (best_candidate, scored_list).
    """
    context = context or {}
    scored = []
    for c in candidates:
        if scorer:
            score = scorer(c, context)
        else:
            # Simple heuristic offline scorer
            score = _default_tot_score(c, problem, context)
        scored.append({"thought": c, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]["thought"] if scored else (candidates[0] if candidates else "")
    return best, scored[:beam]


def lats_search(
    candidates: List[Dict[str, Any]],
    evaluate_fn: Callable[[Dict[str, Any]], float],
    depth: int = 2,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Lightweight LATS-style search: expand candidates, score by a real
    external check (not the model's own opinion), return best path.
    """
    scored = []
    for cand in candidates:
        score = evaluate_fn(cand)
        scored.append({**cand, "lats_score": score})
    scored.sort(key=lambda x: x["lats_score"], reverse=True)
    return (scored[0] if scored else {}), scored


def _default_tot_score(thought: str, problem: str, context: Dict) -> float:
    t = thought.lower()
    score = 0.5
    if "clinical" in t or "medical necessity" in t:
        score += 0.3
    if "guideline" in t or "protocol" in t:
        score += 0.2
    if "cost" in t or "cheapest" in t:
        score -= 0.1
    if context.get("prior_denial_reason") and context["prior_denial_reason"].lower() in t:
        score += 0.25
    return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# 3. Constrained ReAct (whitelist of allowed actions / tools)
# ---------------------------------------------------------------------------

class ConstrainedReAct:
    """
    ReAct loop that may ONLY call tools present in `allowed_tools`.
    Any attempt to call outside the whitelist raises and can become a ticket.
    """

    def __init__(self, allowed_tools: List[str], max_iters: int = 5):
        self.allowed = set(allowed_tools)
        self.max_iters = max_iters
        self.trace: List[Dict[str, Any]] = []

    def step(
        self,
        observation: str,
        proposed_action: str,
        tool_args: Optional[Dict] = None,
        tool_executor: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        if proposed_action not in self.allowed:
            raise PermissionError(
                f"ConstrainedReAct: action '{proposed_action}' not in whitelist {sorted(self.allowed)}"
            )
        result = {"action": proposed_action, "args": tool_args or {}, "ok": True, "output": None}
        if tool_executor:
            try:
                result["output"] = tool_executor(proposed_action, tool_args or {})
            except Exception as e:
                result["ok"] = False
                result["output"] = str(e)
                raise
        self.trace.append({"observation": observation, **result})
        return result

    def run_plan(
        self,
        plan: List[Dict[str, Any]],
        tool_executor: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for step in plan[: self.max_iters]:
            action = step.get("action") or step.get("tool")
            args = step.get("args") or step.get("tool_args") or {}
            obs = step.get("observation", f"executing {action}")
            results.append(self.step(obs, action, args, tool_executor))
        return results


# ---------------------------------------------------------------------------
# 4. RAG (reuse hospital policy / protocol corpus when available)
# ---------------------------------------------------------------------------

def rag_retrieve(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    Retrieve relevant hospital policy / protocol snippets.
    Tries the existing rag/ hybrid index; falls back to embedded stubs so
    graphs work even if the vector store has not been built yet.
    """
    try:
        import sys
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from rag.hybrid_search import build_hybrid_index, hybrid_rag_answer
        # hybrid_rag_answer returns a full answer string; we also want raw hits.
        # Prefer a lightweight fallback that always works.
        answer = hybrid_rag_answer(query) if callable(hybrid_rag_answer) else None
        if answer:
            return [{"source": "hybrid_rag", "text": str(answer)[:800]}]
    except Exception:
        pass

    # Embedded protocol stubs for offline / first-run demos
    stubs = [
        {
            "source": "post_op_protocol_v3",
            "text": (
                "Post-operative recovery protocol: vitals q4h for 48h; "
                "CBC and basic metabolic panel at 6h and 24h; "
                "escalate if HR > 120, SBP < 90, or pain score > 7 sustained; "
                "physician sign-off required before advancing to phase-2 mobility or discharge planning."
            ),
        },
        {
            "source": "insurance_preauth_policy",
            "text": (
                "Elective procedure pre-authorization: submit CPT + ICD-10 + clinical notes "
                "within payer window; if denied, file appeal within 14 days citing medical necessity "
                "and relevant specialty guidelines; never resubmit identical packet after denial."
            ),
        },
        {
            "source": "ed_triage_acuity",
            "text": (
                "ED surge triage: ESI levels 1-5; irreversible actions (sedation, ICU admission, "
                "OR booking) require on-call attending approval; order by acuity then resource "
                "availability; never downgrade ESI without documented reassessment."
            ),
        },
    ]
    q = query.lower()
    ranked = sorted(
        stubs,
        key=lambda s: sum(1 for w in q.split() if w in s["text"].lower()),
        reverse=True,
    )
    return ranked[:top_k]

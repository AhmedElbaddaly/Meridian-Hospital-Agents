"""
planning/tree_of_thoughts.py
------------------------------
Tree of Thoughts: generate several candidate next-steps, self-evaluate
each, keep the best (beam search). Forked from:
github.com/AmrSheta22/task_decomposition_and_planning
    (planning_lab/algorithms/tree_of_thoughts.py)

The toolkit version searches over free-text "partial solution paths" for
an open-ended problem (Game-of-24 style). Our domain problem is narrower
and higher-stakes: `rank_by_urgency` must turn a *classified* patient
list into a *complete ordering*, where the naive "just sort by severity"
answer is not always the best one a lookahead search would find.

Adapted for the ER-surge `rank_by_urgency` node:
  - a "thought" here is not a partial path toward a longer answer, it IS
    a complete candidate ranking. We use ToT's generate -> evaluate ->
    keep-best loop as a *self-consistency / best-of-N with independent
    critique* search over whole rankings, which is the shape this
    sub-task actually needs (see planning/router.py's rationale table).
  - candidates are generated once (breadth), each is scored against an
    explicit rubric that goes beyond "RED before YELLOW before GREEN":
    ties within a severity band should be broken toward whichever
    ordering front-loads patients who are more likely to need a bed
    *right now* (age + which diagnosis keywords look most unstable),
    not sorted arbitrarily. This is exactly the kind of judgment call
    where a single deterministic pass (Plan-and-Solve) can miss a
    defensible-but-worse tie-break that a second candidate + an
    independent evaluator catches.
  - depth is fixed at 1 (one generate/evaluate round) because the real
    sub-task is "pick the best full ranking now", not an iterative
    path-building search across many rounds -- using the toolkit's
    multi-depth loop unmodified here would just multiply LLM calls
    without a task-shape reason, which the lab explicitly penalizes.

Why ToT and not Plan-and-Solve for this node: severity alone is
insufficient information once two+ patients share a severity band --
there IS branching (more than one defensible ordering), and a wrong
tie-break has a real downstream cost (propose_assignment hands out beds
in this exact order). Why ToT and not LATS: there is no real *external*
signal to check candidate rankings against yet -- no bed has been
touched, nothing has been written -- so paying for LATS's MCTS +
grounded environment machinery here would be, in the lab's own words,
"expensive theater". LATS is reserved for `propose_assignment`, the
node that actually touches the real database.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .llm_client import PlanningLLM

GENERATE_SYSTEM = (
    "You generate distinct candidate full orderings for Tree-of-Thoughts search "
    "over an ER patient ranking sub-task. Severity band order (RED, YELLOW, GREEN) "
    "must never be violated; only tie-breaks WITHIN a severity band may differ "
    "between candidates."
)
EVALUATE_SYSTEM = (
    "You independently evaluate one candidate ER patient ranking. Score how well "
    "it prioritizes patients who need a bed soonest within their severity band. "
    "Do not reward confident wording; check the actual order against the rubric."
)

_SEVERITY_ORDER = {"RED": 0, "YELLOW": 1, "GREEN": 2}

# Keywords that make a diagnosis within a band look more unstable / time
# sensitive -- used only by the OFFLINE fallback candidate generator +
# scorer below, so the search is domain-aware rather than a random stub
# (same convention as decomposition.py / plan_and_solve.py).
_URGENCY_HINTS = ("arrest", "bleeding", "trauma", "unresponsive", "collapse")


@dataclass
class Thought:
    """A complete candidate ranking, not a partial path (see module docstring)."""

    state: list[dict]
    score: float
    rationale: str


@dataclass
class ToTResult:
    task_id: str
    frontier: list[Thought]
    best: Thought
    output: str
    llm_calls: int
    input_tokens: int
    output_tokens: int
    latency_s: float


def _candidate_a_severity_then_age(patients: list[dict]) -> list[dict]:
    """Naive baseline candidate: sort by severity, then oldest first.
    This is what a single Plan-and-Solve pass would produce -- ToT's job
    is to show whether a second candidate beats it."""
    return sorted(patients, key=lambda p: (_SEVERITY_ORDER[p["severity"]], -p["age"]))


def _candidate_b_severity_then_instability(patients: list[dict]) -> list[dict]:
    """Second candidate: within a band, front-load diagnoses that look
    most likely to deteriorate in the next few minutes, THEN age."""

    def instability_key(p: dict) -> int:
        text = p["diagnosis"].lower()
        return 0 if any(h in text for h in _URGENCY_HINTS) else 1

    return sorted(
        patients,
        key=lambda p: (_SEVERITY_ORDER[p["severity"]], instability_key(p), -p["age"]),
    )


def _score_ranking(patients: list[dict], ranking: list[dict]) -> tuple[float, str]:
    """Deterministic offline scorer: rewards keeping severity bands intact
    AND front-loading unstable diagnoses within a band. Same role as the
    toolkit's `ThoughtEvaluation`, but grounded in the domain rule instead
    of an opaque model opinion (Person 3's grounded-vs-ungrounded concern
    applies here too: this offline scorer is itself a rubric check, not a
    coin flip)."""
    bands = [_SEVERITY_ORDER[p["severity"]] for p in ranking]
    if bands != sorted(bands):
        return 0.1, "Severity band order violated -- invalid ranking."

    misplacements = 0
    i = 0
    while i < len(ranking):
        band = ranking[i]["severity"]
        j = i
        while j < len(ranking) and ranking[j]["severity"] == band:
            j += 1
        band_slice = ranking[i:j]
        stable_first = sorted(
            band_slice,
            key=lambda p: (0 if any(h in p["diagnosis"].lower() for h in _URGENCY_HINTS) else 1, -p["age"]),
        )
        misplacements += sum(1 for a, b in zip(band_slice, stable_first) if a["name"] != b["name"])
        i = j

    score = max(0.2, 1.0 - 0.2 * misplacements)
    rationale = (
        "Severity bands intact; 0 unstable-diagnosis misplacements within band."
        if misplacements == 0
        else f"Severity bands intact but {misplacements} unstable-diagnosis patient(s) "
        "ranked behind a less time-sensitive same-band patient."
    )
    return score, rationale


def tree_of_thoughts(
    task_id: str,
    goal: str,
    instruction: str,
    llm: PlanningLLM,
    state: dict,
    beam_width: int = 2,
) -> ToTResult:
    """One generate round (up to `beam_width` candidates) + one independent
    evaluate call per candidate, then keep the best. See module docstring
    for why depth is fixed at 1 for this sub-task."""
    t0 = time.time()
    calls_before = llm.call_count
    in_before = llm.total_input_tokens
    out_before = llm.total_output_tokens

    if task_id != "rank_by_urgency":
        raise NotImplementedError(
            f"tree_of_thoughts has no wired candidate generator for task_id={task_id!r}; "
            "route it elsewhere (see planning/router.py)."
        )

    patients = state["classified_patients"]

    def generate_fallback(_prompt: str) -> dict:
        cand_a = _candidate_a_severity_then_age(patients)
        cand_b = _candidate_b_severity_then_instability(patients)
        return {
            "candidates": [
                [p["name"] for p in cand_a],
                [p["name"] for p in cand_b],
            ]
        }

    generated = llm.structured(
        system=GENERATE_SYSTEM,
        user=(
            f"Goal: {goal}\nSub-task: {instruction}\n"
            f"Classified patients: {[(p['name'], p['severity'], p['age'], p['diagnosis']) for p in patients]}\n"
            f"Propose up to {beam_width} distinct full orderings (patient names only)."
        ),
        fallback_fn=generate_fallback,
        label=f"tree_of_thoughts.{task_id}.generate",
    )

    by_name = {p["name"]: p for p in patients}
    candidate_rankings: list[list[dict]] = []
    for name_order in generated["candidates"][:beam_width]:
        try:
            candidate_rankings.append([by_name[n] for n in name_order])
        except KeyError:
            continue  # a malformed online candidate is dropped, not crashed on
    if not candidate_rankings:
        candidate_rankings = [_candidate_a_severity_then_age(patients)]

    frontier: list[Thought] = []
    for ranking in candidate_rankings:

        def evaluate_fallback(_prompt: str, _ranking=ranking) -> dict:
            score, rationale = _score_ranking(patients, _ranking)
            return {"score": score, "rationale": rationale}

        judged = llm.structured(
            system=EVALUATE_SYSTEM,
            user=(
                f"Goal: {goal}\nCandidate ranking: {[p['name'] for p in ranking]}\n"
                f"Patient detail: {[(p['name'], p['severity'], p['age'], p['diagnosis']) for p in ranking]}\n"
                "Score 0-1 how well this front-loads time-sensitive patients within each severity band."
            ),
            fallback_fn=evaluate_fallback,
            label=f"tree_of_thoughts.{task_id}.evaluate",
        )
        frontier.append(Thought(state=ranking, score=judged["score"], rationale=judged["rationale"]))

    frontier.sort(key=lambda t: t.score, reverse=True)
    best = frontier[0]
    state["ranked_patients"] = best.state
    output = "Order: " + ", ".join(f"{p['name']}({p['severity']})" for p in best.state)

    return ToTResult(
        task_id=task_id,
        frontier=frontier,
        best=best,
        output=output,
        llm_calls=llm.call_count - calls_before,
        input_tokens=llm.total_input_tokens - in_before,
        output_tokens=llm.total_output_tokens - out_before,
        latency_s=round(time.time() - t0, 4),
    )

"""
planning/router.py
--------------------
Routing logic: decides which planning algorithm a `kind="reasoning"`
DAG node (planning/models.py's Task) gets executed by. This is the piece
the lab explicitly asks for -- "each sub-task in your DAG that needs
more than a single deterministic tool call should be routed to whichever
of the three actually fits its shape, and you should be able to say why"
-- and is what agent/planning_agent.py (Person 1's integration) calls
instead of hand-picking an algorithm inline.

Only `kind="reasoning"` tasks are routed here. `kind="read"` / `kind="write"`
tasks (check_capacity, apply_and_report) go straight to
planning/mcp_grounding.py -- there is no planning algorithm to route a
single deterministic tool call to, and routing one through PS/ToT/LATS
would itself be exactly the "three fake to-do items" the lab warns
against.

ROUTING TABLE (the reason each choice exists, not which sounds fanciest):

  assess_incoming    -> Plan-and-Solve
    One correct classification per patient against a fixed, published
    rule set (TRIAGE_GUIDELINES). No branching exists to search over;
    ToT/LATS would pay 2-4x the calls to re-derive the same answer.

  rank_by_urgency     -> Tree of Thoughts
    Severity band order is fixed, but ties WITHIN a band are genuinely
    ambiguous (age vs. diagnosis-instability) and the wrong tie-break has
    a real downstream cost (propose_assignment hands out beds in this
    exact order). No real external system has been touched yet, so
    there is nothing for LATS's grounded environment to check against --
    self-evaluated candidates are the right amount of machinery here.

  propose_assignment  -> LATS
    The one reasoning node that touches a REAL external system before
    committing: a candidate assignment can be checked against the real
    free-bed snapshot (planning/lats.py's ICUAssignmentEnvironment) for
    double-booking or non-existent beds. A wrong output here is the
    single most expensive mistake in the whole DAG (an unwound
    double-booked bed, by phone, mid-surge) -- exactly the "real cost of
    a wrong branch" LATS's reflection-on-failure loop is built for.

  validate_assignment -> not routed here
    Person 1's decomposition.py/dynamic_decomposition.py implement this
    node directly as a real DB re-check (mcp_grounding.check_capacity),
    not an LLM planning call -- there is no algorithm choice to make on
    a single deterministic re-check.

See planning_eval/results/planning_results.json (produced by
planning_eval/evaluate_planning.py) for the numbers that back this
table, and planning/README.md for the write-up.
"""

from __future__ import annotations

from dataclasses import dataclass

# id -> (algorithm, one-line reason), also usable for --explain output /
# artifacts/ traces without re-deriving the docstring above at runtime.
ROUTING_TABLE: dict[str, tuple[str, str]] = {
    "assess_incoming": (
        "ps",
        "Single correct answer per patient against a fixed triage rule set; no branching to search over.",
    ),
    "rank_by_urgency": (
        "tot",
        "Severity order is fixed but within-band tie-breaks are genuinely ambiguous and costly if wrong.",
    ),
    "propose_assignment": (
        "lats",
        "Only reasoning node with a real external system to check a candidate against before it commits.",
    ),
}

VALID_ALGORITHMS = {"ps", "tot", "lats"}


@dataclass
class RoutingDecision:
    task_id: str
    algorithm: str
    reason: str


def route_task(task_id: str, task_kind: str = "reasoning") -> RoutingDecision:
    """Return which planning algorithm owns this DAG node.

    Raises for anything not `kind="reasoning"` (read/write nodes are
    executed directly against mcp_grounding.py, see module docstring)
    and for any reasoning task_id not in ROUTING_TABLE -- an unrouted
    reasoning node is a routing gap that should fail loudly at
    integration time, not silently default to one algorithm.
    """
    if task_kind != "reasoning":
        raise ValueError(
            f"route_task() only routes kind='reasoning' nodes, got task_id={task_id!r} kind={task_kind!r}. "
            "read/write nodes execute directly via mcp_grounding.py."
        )
    if task_id not in ROUTING_TABLE:
        raise KeyError(
            f"No routing entry for reasoning task_id={task_id!r}. "
            f"Known: {sorted(ROUTING_TABLE)}. Add an entry (and its rationale) to ROUTING_TABLE."
        )
    algorithm, reason = ROUTING_TABLE[task_id]
    return RoutingDecision(task_id=task_id, algorithm=algorithm, reason=reason)


def run_routed(
    task_id: str,
    goal: str,
    instruction: str,
    llm,
    state: dict,
    task_kind: str = "reasoning",
    **algo_kwargs,
):
    """Convenience dispatcher used by agent/planning_agent.py: routes,
    then actually calls the chosen algorithm module, returning its
    native result object (PlanAndSolveResult / ToTResult / LATSResult)
    so callers keep full detail instead of a lossy common shape."""
    decision = route_task(task_id, task_kind)

    if decision.algorithm == "ps":
        from .plan_and_solve import plan_and_solve

        return decision, plan_and_solve(task_id, goal, instruction, llm, state)

    if decision.algorithm == "tot":
        from .tree_of_thoughts import tree_of_thoughts

        return decision, tree_of_thoughts(task_id, goal, instruction, llm, state, **algo_kwargs)

    if decision.algorithm == "lats":
        from .lats import lats

        return decision, lats(task_id, goal, instruction, llm, state, **algo_kwargs)

    raise AssertionError(f"unreachable: unknown algorithm {decision.algorithm!r}")  # pragma: no cover

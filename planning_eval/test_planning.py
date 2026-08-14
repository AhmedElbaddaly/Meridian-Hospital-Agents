"""
planning_eval/test_planning.py
---------------------------------
Offline, deterministic tests for Person 2's concern (Planning Algorithms
+ Routing). Same convention as Person 1's planning_eval/test_decomposition.py:
no ANTHROPIC_API_KEY required, every case runs against the real
deterministic offline fallbacks in planning/plan_and_solve.py,
planning/tree_of_thoughts.py, and planning/lats.py.

Run: python -m pytest planning_eval/test_planning.py -q
"""

from __future__ import annotations

import pytest

from planning.llm_client import PlanningLLM
from planning.router import route_task, run_routed, ROUTING_TABLE
from planning.plan_and_solve import plan_and_solve
from planning.tree_of_thoughts import tree_of_thoughts, _score_ranking
from planning.lats import lats, ICUAssignmentEnvironment


@pytest.fixture
def llm() -> PlanningLLM:
    return PlanningLLM()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def test_router_maps_every_reasoning_node():
    assert route_task("assess_incoming").algorithm == "ps"
    assert route_task("rank_by_urgency").algorithm == "tot"
    assert route_task("propose_assignment").algorithm == "lats"


def test_router_rejects_non_reasoning_kind():
    with pytest.raises(ValueError):
        route_task("check_capacity", task_kind="read")


def test_router_rejects_unknown_task_id():
    with pytest.raises(KeyError):
        route_task("some_future_node")


def test_routing_table_has_a_reason_for_every_entry():
    for task_id, (algorithm, reason) in ROUTING_TABLE.items():
        assert algorithm in {"ps", "tot", "lats"}
        assert len(reason) > 20, f"{task_id} routing reason looks like a placeholder"


# ---------------------------------------------------------------------------
# Plan-and-Solve
# ---------------------------------------------------------------------------
def test_plan_and_solve_classifies_correctly(llm):
    state = {
        "incoming_patients": [
            {"name": "Amina", "age": 34, "diagnosis": "Blunt trauma, chest"},
            {"name": "Karim", "age": 61, "diagnosis": "Abdominal pain, fever"},
            {"name": "Layla", "age": 22, "diagnosis": "Minor laceration"},
        ]
    }
    result = plan_and_solve("assess_incoming", "ER surge reshuffle", "Classify urgency.", llm, state)
    got = {p["name"]: p["severity"] for p in state["classified_patients"]}
    assert got == {"Amina": "RED", "Karim": "YELLOW", "Layla": "GREEN"}
    assert result.llm_calls == 2  # one plan call, one solve call -- the concern's "single pass, no branching"


def test_plan_and_solve_rejects_unrouted_task(llm):
    with pytest.raises(NotImplementedError):
        plan_and_solve("propose_assignment", "goal", "instr", llm, {"incoming_patients": []})


# ---------------------------------------------------------------------------
# Tree of Thoughts
# ---------------------------------------------------------------------------
def test_tot_prefers_instability_aware_ranking_over_naive_age_sort(llm):
    """The required divergence case: naive age-first tie-break is WRONG,
    ToT's second candidate is right, and ToT picks the right one."""
    patients = [
        {"name": "Amina", "age": 34, "diagnosis": "Blunt trauma, chest, arrest risk", "severity": "RED"},
        {"name": "Youssef", "age": 70, "diagnosis": "Cardiac chest pain, stable", "severity": "RED"},
    ]
    state = {"classified_patients": patients}
    result = tree_of_thoughts("rank_by_urgency", "goal", "Order patients.", llm, state)
    assert [p["name"] for p in result.best.state] == ["Amina", "Youssef"]
    assert result.best.score > sorted(t.score for t in result.frontier)[0]  # best beats at least one alternative


def test_tot_never_violates_severity_band_order(llm):
    patients = [
        {"name": "A", "age": 20, "diagnosis": "Fever", "severity": "GREEN"},
        {"name": "B", "age": 90, "diagnosis": "Cardiac arrest", "severity": "RED"},
    ]
    state = {"classified_patients": patients}
    result = tree_of_thoughts("rank_by_urgency", "goal", "Order patients.", llm, state)
    severities = [p["severity"] for p in result.best.state]
    assert severities == ["RED", "GREEN"]


def test_scoring_rubric_rejects_band_order_violation():
    patients = [{"name": "A", "severity": "GREEN", "age": 10, "diagnosis": "x"}, {"name": "B", "severity": "RED", "age": 10, "diagnosis": "y"}]
    bad_ranking = patients  # GREEN before RED -- invalid
    score, reason = _score_ranking(patients, bad_ranking)
    assert score < 0.5
    assert "violated" in reason.lower()


# ---------------------------------------------------------------------------
# LATS
# ---------------------------------------------------------------------------
def test_lats_grounded_environment_catches_double_booking_real_check():
    """The required grounded-vs-ungrounded evidence: a candidate that
    reuses a bed id must fail, using ONLY the real free-bed data, no
    model opinion involved."""
    env = ICUAssignmentEnvironment()
    feedback = env.evaluate({"Amina": 8, "Youssef": 8})  # same bed twice
    assert feedback.success is False
    assert any("double-book" in d.lower() for d in feedback.details)


def test_lats_grounded_environment_catches_bed_not_actually_free():
    env = ICUAssignmentEnvironment()
    feedback = env.evaluate({"Amina": 9999})  # bed id that doesn't exist in the real snapshot
    assert feedback.success is False
    assert any("not present in the real free-bed snapshot" in d.lower() for d in feedback.details)


def test_lats_recovers_from_a_failed_branch_via_reflection(llm):
    """The required LATS demo: iteration 1's naive action fails a REAL
    check, gets a grounded reflection, iteration/child 2 succeeds within
    the same run."""
    state = {
        "ranked_patients": [
            {"name": "Nadia", "age": 45, "diagnosis": "Severe bleeding", "severity": "RED"},
            {"name": "Omar", "age": 29, "diagnosis": "Blunt trauma", "severity": "RED"},
            {"name": "Huda", "age": 52, "diagnosis": "Cardiac arrest", "severity": "RED"},
        ],
        "free_beds_snapshot": [{"bed_id": 8, "room": "ICU-08"}, {"bed_id": 9, "room": "ICU-09"}],
    }
    result = lats("propose_assignment", "goal", "instr", llm, state, iterations=2, n_actions=2)
    assert result.success is True
    # the first child in the tree must be the deliberately flawed one, and
    # it must carry a reflection -- proves the failure->reflection link,
    # not just a lucky first try.
    failed_children = [c for c in result.root.children if c.feedback and not c.feedback.success]
    assert failed_children, "expected at least one genuinely rejected branch in this scenario"
    assert failed_children[0].reflections, "a rejected branch must carry a grounded reflection"


def test_lats_never_hallucinates_a_bed_that_doesnt_exist():
    """Overflow scenario: fewer real beds than RED patients. LATS must
    not invent a bed id to force success."""
    llm = PlanningLLM()
    state = {
        "ranked_patients": [
            {"name": "Sami", "age": 61, "diagnosis": "Trauma", "severity": "RED"},
            {"name": "Dina", "age": 33, "diagnosis": "Cardiac arrest", "severity": "RED"},
        ],
        "free_beds_snapshot": [{"bed_id": 8, "room": "ICU-08"}],
    }
    result = lats("propose_assignment", "goal", "instr", llm, state, iterations=2, n_actions=2)
    assigned_beds = list(result.best_candidate.values())
    assert set(assigned_beds).issubset({8}), "must only ever use real, existing bed ids"


def test_lats_rejects_unrouted_task():
    llm = PlanningLLM()
    with pytest.raises(NotImplementedError):
        lats("assess_incoming", "goal", "instr", llm, {})


# ---------------------------------------------------------------------------
# End-to-end via the router (mirrors what agent/planning_agent.py will do)
# ---------------------------------------------------------------------------
def test_router_chains_all_three_algorithms_across_the_dag(llm):
    state = {
        "incoming_patients": [
            {"name": "Amina", "age": 34, "diagnosis": "Blunt trauma, chest, arrest risk"},
            {"name": "Youssef", "age": 70, "diagnosis": "Cardiac chest pain, stable"},
        ],
        "free_beds_snapshot": [{"bed_id": 8, "room": "ICU-08"}, {"bed_id": 9, "room": "ICU-09"}],
    }
    _, r1 = run_routed("assess_incoming", "goal", "Classify urgency.", llm, state)
    assert state["classified_patients"]

    _, r2 = run_routed("rank_by_urgency", "goal", "Order patients.", llm, state)
    assert state["ranked_patients"]

    _, r3 = run_routed("propose_assignment", "goal", "Assign beds.", llm, state)
    assert state["assignments"]

"""
planning_eval/test_cases/planning_cases.py
--------------------------------------------
FROZEN test suite for Person 2's concern (Planning Algorithms + Routing).
Per the team rule ("the test suite freezes in Phase 0 before any
evaluation"), do not edit case contents after evaluate_planning.py has
been run for the comparison table -- add new cases in a new function
instead.

Three groups, matching the three routed nodes:

  ASSESS_CASES   -- assess_incoming, run through Plan-and-Solve
  RANK_CASES     -- rank_by_urgency, run through Tree of Thoughts
                    AND a naive single-pass baseline (to make the PS-vs-ToT
                    comparison meaningful for this node, since the lab
                    requires comparing "every applicable case", not only
                    the node each algorithm ended up owning)
  ASSIGN_CASES   -- propose_assignment, run through LATS
                    AND a naive first-fit baseline (same reasoning)

Each RANK/ASSIGN case documents, in its own comment, whether it is a
case where the cheap baseline already gets it right (so the fancier
search costs more for nothing) or a case where it genuinely needs the
extra machinery to catch something -- both kinds are required by the lab
("include at least one case that should favor X... and one that needs
lookahead search").
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# ASSESS_CASES: assess_incoming (Plan-and-Solve)
# ---------------------------------------------------------------------------
ASSESS_CASES = [
    {
        "name": "A1_clear_mixed_severity",
        "incoming_patients": [
            {"name": "Amina", "age": 34, "diagnosis": "Blunt trauma, chest, arrest risk"},
            {"name": "Karim", "age": 61, "diagnosis": "Abdominal pain, fever"},
            {"name": "Layla", "age": 22, "diagnosis": "Minor laceration"},
        ],
        "expected_severity": {"Amina": "RED", "Karim": "YELLOW", "Layla": "GREEN"},
    },
    {
        "name": "A2_all_red_mass_casualty",
        "incoming_patients": [
            {"name": "Youssef", "age": 70, "diagnosis": "Cardiac arrest"},
            {"name": "Nadia", "age": 45, "diagnosis": "Severe bleeding"},
            {"name": "Omar", "age": 29, "diagnosis": "Blunt trauma"},
        ],
        "expected_severity": {"Youssef": "RED", "Nadia": "RED", "Omar": "RED"},
    },
    {
        "name": "A3_single_patient_walkin",
        "incoming_patients": [
            {"name": "Hana", "age": 8, "diagnosis": "Asthma flare-up"},
        ],
        "expected_severity": {"Hana": "YELLOW"},
    },
]

# ---------------------------------------------------------------------------
# RANK_CASES: rank_by_urgency (Tree of Thoughts vs. naive-single-pass baseline)
# ---------------------------------------------------------------------------
RANK_CASES = [
    {
        # Favors ToT: naive age-first tie-break puts the STABLE 70-year-old
        # ahead of the arrest-risk 34-year-old within the same RED band --
        # a real ordering mistake the second, instability-aware candidate
        # catches and the evaluator scores strictly higher.
        "name": "R1_diverges_favor_tot",
        "classified_patients": [
            {"name": "Amina", "age": 34, "diagnosis": "Blunt trauma, chest, arrest risk", "severity": "RED"},
            {"name": "Youssef", "age": 70, "diagnosis": "Cardiac chest pain, stable", "severity": "RED"},
            {"name": "Karim", "age": 61, "diagnosis": "Abdominal pain, fever", "severity": "YELLOW"},
        ],
        "diverges": True,
    },
    {
        # Favors the naive baseline: age-first and instability-first agree
        # here (the older patient IS also the unstable one), so ToT's extra
        # generate+evaluate calls buy nothing over the cheap single pass.
        "name": "R2_naive_already_correct",
        "classified_patients": [
            {"name": "Fatima", "age": 80, "diagnosis": "Cardiac arrest", "severity": "RED"},
            {"name": "Tarek", "age": 30, "diagnosis": "Stable chest pain, monitored", "severity": "RED"},
            {"name": "Sara", "age": 50, "diagnosis": "Fever", "severity": "YELLOW"},
        ],
        "diverges": False,
    },
    {
        # Larger band with 3 same-severity patients and 2 of them unstable --
        # genuinely needs the search to get every tie-break right, not just one.
        "name": "R3_multi_tie_within_band",
        "classified_patients": [
            {"name": "Nadia", "age": 45, "diagnosis": "Severe bleeding", "severity": "RED"},
            {"name": "Zaid", "age": 68, "diagnosis": "Cardiac chest pain, stable", "severity": "RED"},
            {"name": "Mona", "age": 40, "diagnosis": "Unresponsive, collapse", "severity": "RED"},
            {"name": "Rami", "age": 55, "diagnosis": "Abdominal pain, fever", "severity": "YELLOW"},
        ],
        "diverges": True,
    },
]

# ---------------------------------------------------------------------------
# ASSIGN_CASES: propose_assignment (LATS vs. naive-first-fit baseline)
# ---------------------------------------------------------------------------
ASSIGN_CASES = [
    {
        # Favors the naive baseline: exactly enough real free beds, no
        # race, first-fit is already valid -- LATS pays for a search that
        # changes nothing.
        "name": "L1_sufficient_capacity",
        "ranked_patients": [
            {"name": "Amina", "age": 34, "diagnosis": "Blunt trauma", "severity": "RED"},
            {"name": "Youssef", "age": 70, "diagnosis": "Cardiac chest pain", "severity": "RED"},
        ],
        "free_beds_snapshot": [{"bed_id": 8, "room": "ICU-08"}, {"bed_id": 9, "room": "ICU-09"}],
    },
    {
        # Favors LATS: this is the exact scenario planning/lats.py's
        # offline fallback is built around -- the naive first-fit action
        # double-books a bed id, only a grounded environment check catches
        # it, and only LATS's reflect+retry produces a valid second
        # candidate within the same run.
        "name": "L2_naive_double_books_needs_lats",
        "ranked_patients": [
            {"name": "Nadia", "age": 45, "diagnosis": "Severe bleeding", "severity": "RED"},
            {"name": "Omar", "age": 29, "diagnosis": "Blunt trauma", "severity": "RED"},
            {"name": "Huda", "age": 52, "diagnosis": "Cardiac arrest", "severity": "RED"},
        ],
        "free_beds_snapshot": [{"bed_id": 8, "room": "ICU-08"}, {"bed_id": 9, "room": "ICU-09"}],
    },
    {
        # Mass-casualty overflow: even LATS cannot invent a bed that
        # doesn't exist -- this case shows LATS correctly reports overflow
        # rather than hallucinating a bed id, matching Person 1's
        # escalate_overflow divergence story instead of contradicting it.
        "name": "L3_overflow_no_hallucinated_beds",
        "ranked_patients": [
            {"name": "Sami", "age": 61, "diagnosis": "Trauma", "severity": "RED"},
            {"name": "Dina", "age": 33, "diagnosis": "Cardiac arrest", "severity": "RED"},
            {"name": "Adam", "age": 19, "diagnosis": "Severe bleeding", "severity": "RED"},
            {"name": "Rana", "age": 77, "diagnosis": "Unresponsive", "severity": "RED"},
        ],
        "free_beds_snapshot": [{"bed_id": 8, "room": "ICU-08"}],
    },
]

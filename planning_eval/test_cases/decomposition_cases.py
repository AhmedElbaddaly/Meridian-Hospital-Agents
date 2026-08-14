"""
planning_eval/test_cases/decomposition_cases.py
--------------------------------------------------
Real request prompts for the "ER surge reshuffle" problem, used to
exercise decomposition-first and dynamic decomposition against the same
request type (required by the lab). Frozen once evaluation starts --
do not edit case data after `planning_eval/evaluate.py` has been run for
the comparison table.

Each case is a (goal, incoming_patients) pair. incoming_patients mimics
patients who just arrived and are NOT YET in the Patients table -- that
registration only happens at the DAG's terminal apply_and_report node,
which is exactly the point of having a terminal write node at all.
"""

# Case A: capacity is sufficient -- decomposition-first and dynamic
# should reach the SAME outcome, just at different cost. This is the
# case that should favor decomposition-first (cheaper, same result).
CASE_A_SUFFICIENT_CAPACITY = (
    "Two patients just arrived by ambulance, reshuffle the ER board now.",
    [
        {"name": "Test Patient A1", "age": 45, "gender": "Male", "diagnosis": "Minor Trauma"},
        {"name": "Test Patient A2", "age": 30, "gender": "Female", "diagnosis": "Mild Fever"},
    ],
)

# Case B: three RED-severity patients vs. two real free ICU beds
# (ICU-08, ICU-09 in db/seed.sql) -- a genuine shortfall. This is the
# case that should favor dynamic decomposition (see divergence_case.py
# for the full side-by-side).
CASE_B_MASS_CASUALTY_SHORTFALL = (
    "Multi-vehicle accident: three critical trauma patients incoming, reshuffle now.",
    [
        {"name": "Test Patient B1", "age": 52, "gender": "Male", "diagnosis": "Severe Trauma - Cardiac Arrest"},
        {"name": "Test Patient B2", "age": 34, "gender": "Female", "diagnosis": "Internal Bleeding Trauma"},
        {"name": "Test Patient B3", "age": 61, "gender": "Male", "diagnosis": "Severe Trauma"},
    ],
)

# Case C: mixed severities -- exercises the ranking sub-task with a real
# ordering decision (RED before YELLOW before GREEN, ties by age).
CASE_C_MIXED_SEVERITY = (
    "Walk-in surge after a building fire: mixed injuries, reshuffle the board.",
    [
        {"name": "Test Patient C1", "age": 70, "gender": "Male", "diagnosis": "Smoke Inhalation Cardiac Arrest"},
        {"name": "Test Patient C2", "age": 25, "gender": "Female", "diagnosis": "Acute Abdominal Pain"},
        {"name": "Test Patient C3", "age": 8, "gender": "Male", "diagnosis": "Minor Laceration"},
    ],
)

ALL_CASES = {
    "A_sufficient_capacity": CASE_A_SUFFICIENT_CAPACITY,
    "B_mass_casualty_shortfall": CASE_B_MASS_CASUALTY_SHORTFALL,
    "C_mixed_severity": CASE_C_MIXED_SEVERITY,
}

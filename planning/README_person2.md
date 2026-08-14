# Planning Lab — Person 2: Planning Algorithms + Routing

## What's here

| File | Concern |
|---|---|
| `plan_and_solve.py` | **Plan-and-Solve**: one explicit plan call, one solve call, no branching. Forked from the toolkit's `algorithms/plan_and_solve.py`, ported onto `PlanningLLM`. Routed to `assess_incoming`. |
| `tree_of_thoughts.py` | **Tree of Thoughts**: generate candidate full rankings, self-evaluate each against a grounded rubric, keep the best. Forked from `algorithms/tree_of_thoughts.py`. Routed to `rank_by_urgency`. |
| `lats.py` | **LATS**: MCTS-lite search (select via UCT, expand + real environment check, backpropagate, reflect on failure). Forked from `algorithms/lats.py`. Routed to `propose_assignment`. Includes `ICUAssignmentEnvironment`, a real read-only grounded check against `mcp_grounding.check_capacity()` — a stand-in for Person 3's `planning/environment.py`, built to the same `EnvironmentFeedback` shape so it swaps in with a one-line change at integration. |
| `router.py` | Central `route_task()` / `run_routed()` — maps each reasoning-kind DAG node to the algorithm above, with the rationale written as code, not just prose. |

## Why these three algorithms don't overlap on this DAG

| Node | Algorithm | Real reason (not "sounds sophisticated") |
|---|---|---|
| `assess_incoming` | Plan-and-Solve | One correct triage classification per patient against a published rule set. No branching exists to search over — ToT/LATS would just re-derive the same answer for 2-4x the calls. |
| `rank_by_urgency` | Tree of Thoughts | Severity band order is fixed, but tie-breaks *within* a band are genuinely ambiguous (age vs. diagnosis-instability), and a bad tie-break directly changes who gets a bed next. Nothing external has been touched yet, so LATS's grounded-environment machinery would have nothing real to check against. |
| `propose_assignment` | LATS | The one reasoning node that can be checked against a REAL system (the actual free-bed snapshot) before it commits. This is where a wrong output is genuinely expensive (a double-booked bed mid-surge) — exactly the case LATS's reflect-on-failure loop exists for. |

## The required divergence demos

**PS vs ToT (rank_by_urgency)** — `planning_eval/test_cases/planning_cases.py::R1_diverges_favor_tot`:
a naive single deterministic pass (severity, then age) ranks the *stable* 70-year-old ahead of
the *arrest-risk* 34-year-old within the same RED band. ToT generates a second, instability-aware
candidate, an independent evaluator scores it strictly higher, and it's the one that ships.

**Naive-first-fit vs LATS (propose_assignment)** — `L2_naive_double_books_needs_lats`:
the naive first-fit action reuses bed `9` for two different patients. `ICUAssignmentEnvironment`
(a real check against the live free-bed snapshot, no model opinion involved) rejects it. LATS
turns that rejection into a grounded reflection ("I proposed... which failed a real check:
double-books bed id(s)...") and the next candidate in the same run succeeds.

## Cost measured, not assumed

`python -m planning_eval.evaluate_planning` runs every method against the frozen 9-case suite
(3 assess, 3 rank, 3 assign) and writes `planning_eval/results/planning_results.json`:

| Method | Task success | Avg. LLM calls | Avg. tokens | Avg. latency |
|---|---|---|---|---|
| Plan-and-Solve (assess_incoming) | 3/3 | 2 | ~156 | ~0.00003s |
| Naive single-pass (rank_by_urgency baseline) | 1/3 | 0 | 0 | 0s |
| Tree of Thoughts (rank_by_urgency) | 3/3 | 3 | ~269 | ~0.0001s |
| Naive first-fit (propose_assignment baseline) | 1/3 | 0 | 0 | 0s |
| LATS (propose_assignment) | 3/3 | 3.3 | ~157 | ~0.0001s |

**Reading the table honestly:** the naive baselines are free and *sometimes* right — for
`rank_by_urgency`, the naive pass already agrees with the grounded rubric whenever the oldest
patient in a band happens to also be the least stable (`R2_naive_already_correct`), so paying
for ToT there buys nothing. But across the 3-case suite the naive pass is wrong 2/3 of the time
in ways that matter (a real bed-priority mistake, or a double-booked bed), while ToT/LATS are
right 3/3 for a modest, bounded extra cost (2-3 extra calls, well under 300 tokens). That's the
actual justification for routing these two nodes through search instead of a single pass — the
table drove it, not a preference for LATS "sounding" more sophisticated. `assess_incoming` has no
naive-vs-PS row because PS *is* the cheap single-pass baseline for that node; there was nothing
cheaper to compare it against.

## Tests

`python -m pytest planning_eval/test_planning.py -q` — 15 tests, fully offline: router validation
(every reasoning node mapped, non-reasoning nodes rejected, unknown nodes rejected loudly),
Plan-and-Solve correctness, ToT's severity-band-order invariant and instability-aware tie-break,
LATS's grounded environment rejecting both a double-booked bed and a nonexistent bed id using only
real data, LATS's reflection-driven recovery from a failed branch within the same run, LATS's
guarantee that it never hallucinates a bed id under overflow, and a full router-chained run across
all three algorithms in one DAG execution.

## Running this piece standalone

```bash
cd Meridian-General-Hospital-Agents-phase2-main
pip install -r requirements.txt
pip install pydantic networkx pytest anthropic   # planning/ additions (same as Person 1)

python -m planning_eval.evaluate_planning
python -m pytest planning_eval/test_planning.py -q
```

No `ANTHROPIC_API_KEY` required — every LLM-shaped step has a documented, deterministic,
domain-aware offline fallback (see `llm_client.py`, same team convention Person 1 established).
Set the key to route PS/ToT/LATS decisions through Claude instead.

## Known integration point for Person 1 / Person 3

- `agent/planning_agent.py` should call `planning/router.py::run_routed()` for every
  `kind="reasoning"` node instead of hand-picking an algorithm, and should execute
  `kind="read"`/`kind="write"` nodes directly via `mcp_grounding.py`, matching how
  `decomposition.py` already dispatches.
- `lats.py`'s `ICUAssignmentEnvironment` is a temporary, genuinely-grounded stand-in for Person
  3's `planning/environment.py::Environment`. Both return `planning/models.py::EnvironmentFeedback`,
  so swapping is a constructor-argument change (`lats(..., environment=<Person 3's env>)`), not a
  rewrite.
